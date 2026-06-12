import numpy as np


mla_row_size: int = 16
mla_num_tiles: int = 100
mla_max_num_rows_per_tile: int = 4096
mla_max_num_rows: int = mla_max_num_rows_per_tile * mla_num_tiles


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def ceil_div_row(x: int) -> int:
    return ceil_div(x, mla_row_size)


def round_up_to(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def round_up_to_row(x: int) -> int:
    return ceil_div_row(x) * mla_row_size


def _rope_scaling_llama3(
    inv_freq: np.ndarray, factor: float, low_freq_factor: float, high_freq_factor: float,
    original_max_position_embeddings: int
) -> np.ndarray:
    """
    Computes LLAMA3 RoPE scaling, which is "llama3" for rope_type in HuggingFace.

    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L385.

    Args:
        inv_freq: Precomputed RoPE inverse frequencies without scaling.
        factor: The scaling factor. The new context length is the original context length
            times the scaling factor.
        low_freq_factor: The scaling factor for low frequencies.
        high_freq_factor: The scaling factor for high frequencies.
        original_max_position_embeddings: The context length used in training.

    Returns:
        Scaled inverse frequencies for the RoPE embeddings.
    """
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor

    wavelen = 2 * np.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (
        original_max_position_embeddings / wavelen - low_freq_factor
    ) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (
        (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    )
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def _rope_scaling_longrope(
    inv_freq: np.ndarray, long_factor: list[float], short_factor: list[float],
    max_position_embeddings: int, original_max_position_embeddings: int
) -> tuple[np.ndarray, float]:
    """
    Computes "longrope" scaling for RoPE.

    The scaling factor is computed based on "max_position_embeddings" in inference and
    "original_max_position_embeddings" in pretraining.

    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L322.

    Args:
        inv_freq: Precomputed RoPE inverse frequencies without scaling.
        long_factor: The scaling factor for long context length.
        short_factor: The scaling factor for short context length.
        max_position_embeddings: The context length used in inferencing.
        original_max_position_embeddings: The context length used in training.

    Returns:
        Scaled inverse frequencies for the RoPE embeddings and post-processing scaling factor
        applied to the computed cos/sin.  The inverse frequencies have the same length as
        long_factor or short_factor.
    """
    n_freqs = len(inv_freq)
    assert len(long_factor) <= n_freqs
    assert len(short_factor) <= n_freqs

    factor = max_position_embeddings / original_max_position_embeddings
    if factor <= 1.0:
        attention_factor = 1.0
    else:
        attention_factor = np.sqrt(1 + np.log(factor) / np.log(original_max_position_embeddings))

    if max_position_embeddings > original_max_position_embeddings:
        ext_factors = np.array(long_factor, dtype=np.float32)
    else:
        ext_factors = np.array(short_factor, dtype=np.float32)
    inv_freq = inv_freq[:len(ext_factors)] / ext_factors

    return inv_freq, attention_factor


def calc_freq_real_imag(
    max_num_tokens: int, rope_type: str, theta: float, head_dim: int, scaling_cfg: dict,
    idx_base: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    inv_freq = inv_freq.astype(np.float32)
    factor = scaling_cfg.get("factor")
    match rope_type:
        case "" | "default":
            pass
        case "linear":
            inv_freq /= factor
        case "llama3":
            inv_freq = _rope_scaling_llama3(
                inv_freq, factor,
                scaling_cfg.get("low_freq_factor"),
                scaling_cfg.get("high_freq_factor"),
                scaling_cfg.get("original_max_position_embeddings")
            )
        case "longrope":
            inv_freq, _ = _rope_scaling_longrope(
                inv_freq,
                scaling_cfg.get("long_factor"),
                scaling_cfg.get("short_factor"),
                max_num_tokens + idx_base,
                scaling_cfg.get("original_max_position_embeddings")
            )
        case _:
            raise ValueError(f"{rope_type} rope type is not supported.")

    n_scaled_freq = len(inv_freq)
    t = np.arange(idx_base, max_num_tokens + idx_base, dtype=np.float32)
    inv_freq = np.outer(inv_freq, t).reshape(n_scaled_freq, max_num_tokens)

    # Return in HWC layout. Perform a copy so that the last dimension is contiguous.
    inv_freq = np.expand_dims(inv_freq.transpose(1, 0), axis=(0, 1)).copy()
    re = np.cos(inv_freq, dtype=np.float64).astype(np.float32)
    im = np.sin(inv_freq, dtype=np.float64).astype(np.float32)

    # Insert 1 for indices that are not scaled
    if n_scaled_freq < head_dim // 2:
        n_unscaled_freq = head_dim // 2 - n_scaled_freq
        re = np.pad(re, ((0, 0), (0, 0), (0, 0), (0, n_unscaled_freq)), constant_values=1)
        im = np.pad(im, ((0, 0), (0, 0), (0, 0), (0, n_unscaled_freq)), constant_values=1)
    return re, im


class MRopeCalc:
    """
    Manages the creation and updating of Multimodal RoPE tables for Qwen2.5-VL and Qwen3-VL.
    """
    def __init__(self, config, image_token_id: int):
        """
        Initializes the calculator and creates the default master RoPE table for text-only input.
        """
        self.config = config
        self.max_seq_len = config.pipeline_cfg.max_num_tokens
        self.image_token_id = image_token_id
        self.last_prompt_text = True
        self._vision_cfg = config.vm_cfg
        self._text_cfg = config.lm_cfg
        rope_scaling_cfg = getattr(self._text_cfg.rope_cfg, "rope_scaling", None)
        self._mrope_sections = getattr(rope_scaling_cfg, "mrope_section", None) or []
        self._mrope_interleaved = bool(getattr(rope_scaling_cfg, "mrope_interleaved", False))
        self._rope_theta = self._text_cfg.rope_cfg.rope_theta
        self._head_dim = self._text_cfg.attn_cfg.head_dim
        self.master_freq_real, self.master_freq_imag = self._calculate_for_text_only()

    def calculate(self, input_ids: np.ndarray, image_grid_thw: list) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Calculates the final freq_real and freq_imag tables for Qwen2.5-VL's Multimodal RoPE.
        Returns half-width tables (complex phasors) for use with complex multiplication.
        """
        thw_freqs, next_pos_id = self._compute_thw_freqs(input_ids, image_grid_thw)

        if self._mrope_sections:
            split_indices = np.cumsum(self._mrope_sections)[:-1]
            freqs_chunks = np.split(thw_freqs, split_indices, axis=-1)
            num_chunks = len(self._mrope_sections)
            # Select the appropriate frequency source (T, H, or W) for each section
            final_freqs_chunks = [freqs_chunks[i][i % 3] for i in range(num_chunks)]
            final_freqs = np.concatenate(final_freqs_chunks, axis=-1)
        else:
            final_freqs = thw_freqs[0]

        freq_real = np.cos(final_freqs)
        freq_imag = np.sin(final_freqs)

        return freq_real.astype(np.float32), freq_imag.astype(np.float32), next_pos_id

    def calculate_interleaved(
        self, input_ids: np.ndarray, image_grid_thw: list
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Calculates the final freq_real and freq_imag tables for models that use interleaved m-RoPE
        layouts (e.g., Qwen3-VL). Returns half-width tables.
        """
        thw_freqs, next_pos_id = self._compute_thw_freqs(input_ids, image_grid_thw)
        interleaved_freqs = self._apply_interleaved_mrope(thw_freqs)
        
        freq_real = np.cos(interleaved_freqs)
        freq_imag = np.sin(interleaved_freqs)
        return freq_real.astype(np.float32), freq_imag.astype(np.float32), next_pos_id

    def _compute_thw_freqs(self, input_ids: np.ndarray, image_grid_thw: list) -> tuple[np.ndarray, int]:
        """
        Computes 3D RoPE frequencies for sequences containing text and vision tokens.
        
        Vision tokens use 3D position IDs (temporal, height, width) while text tokens
        use the same position ID repeated across all 3 dimensions.
        
        Args:
            input_ids: Token IDs including image placeholder tokens
            image_grid_thw: List of (t, h, w) tuples for each image's grid dimensions
            
        Returns:
            Tuple of (thw_freqs, next_pos_id) where:
            - thw_freqs: Shape (3, seq_len, head_dim//2) containing frequency tables
                         for temporal, height, and width dimensions
            - next_pos_id: Next available position ID for decoding
        """
        spatial_merge_size = self._vision_cfg.spatial_merge_size
        image_token_id = self.image_token_id
        seq_len = len(input_ids)

        llm_pos_ids_list = []
        current_pos_idx = 0
        search_start_idx = 0
        ids_seq_list = input_ids

        for image_idx, (t, h, w) in enumerate(image_grid_thw):
            try:
                image_placeholder_idx = ids_seq_list.index(image_token_id, search_start_idx)
            except ValueError:
                raise RuntimeError("Mismatch between image_grid_thw and image tokens in input_ids.")

            text_len = image_placeholder_idx - search_start_idx
            if text_len > 0:
                text_positions = np.arange(current_pos_idx, current_pos_idx + text_len)
                llm_pos_ids_list.append(np.tile(text_positions.reshape(1, -1), (3, 1)))
                current_pos_idx += text_len

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size
            num_vision_tokens_for_image = llm_grid_t * llm_grid_h * llm_grid_w

            t_index = np.repeat(np.arange(llm_grid_t), llm_grid_h * llm_grid_w)
            h_index = np.tile(np.repeat(np.arange(llm_grid_h), llm_grid_w), llm_grid_t)
            w_index = np.tile(np.arange(llm_grid_w), llm_grid_t * llm_grid_h)

            vision_positions = np.stack([t_index, h_index, w_index]) + current_pos_idx
            llm_pos_ids_list.append(vision_positions)

            current_pos_idx = int(vision_positions.max()) + 1
            search_start_idx = image_placeholder_idx + num_vision_tokens_for_image

        final_text_len = len(ids_seq_list) - search_start_idx
        if final_text_len > 0:
            text_positions = np.arange(current_pos_idx, current_pos_idx + final_text_len)
            llm_pos_ids_list.append(np.tile(text_positions.reshape(1, -1), (3, 1)))

        next_pos_id = current_pos_idx + final_text_len

        position_ids_3d = np.concatenate(llm_pos_ids_list, axis=1)
        final_seq_len = position_ids_3d.shape[1]

        inv_freq = 1.0 / (self._rope_theta ** (np.arange(0, self._head_dim, 2, dtype=np.float32) / self._head_dim))
        thw_freqs = np.einsum("ds,h->dsh", position_ids_3d.astype(np.float32), inv_freq)
        return thw_freqs, next_pos_id

    def _apply_interleaved_mrope(self, freqs: np.ndarray) -> np.ndarray:
        """
        Applies interleaved multimodal RoPE by mixing temporal/height/width frequencies.
        
        Instead of concatenating T, H, W frequencies (as in Qwen2.5-VL), this interleaves
        them in a pattern: [T0, H0, W0, T1, H1, W1, ...] using strided slicing.
        
        Args:
            freqs: Shape (3, seq_len, head_dim) array with [T, H, W] frequency tables
            
        Returns:
            Interleaved frequency table of shape (seq_len, head_dim)
        """
        if freqs.shape[0] < 3 or len(self._mrope_sections) < 3:
            return freqs[0].copy()

        freqs_t = np.copy(freqs[0])
        for dim in (1, 2):
            length = self._mrope_sections[dim] * 3
            length = min(length, freqs_t.shape[-1])
            idx = slice(dim, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
    
    def _calculate_for_text_only(self):
        """Calculates the RoPE tables for a pure text sequence up to max_seq_len."""
        dummy_input_ids = np.arange(self.max_seq_len).tolist()
        empty_image_grid = []

        if self._mrope_interleaved:
            freq_real, freq_imag, _ = self.calculate_interleaved(
                input_ids=dummy_input_ids,
                image_grid_thw=empty_image_grid
            )
        else:
            freq_real, freq_imag, _ = self.calculate(
                input_ids=dummy_input_ids,
                image_grid_thw=empty_image_grid
            )
        return freq_real, freq_imag

    def update_mrope_tables_for_prefill(self, input_ids, image_grid_thw):
        """
        Calculates a full-sized RoPE table for a prefill phase. If images are
        present, it computes the prompt-specific values and merges them with the
        master table for subsequent decoding.
        """
        # If there are no images, the master table is already correct. Return it directly.
        if image_grid_thw is None:
            if self.last_prompt_text:
                return None, None  # No update needed if last was also text
            else:
                self.last_prompt_text = True
                return self.master_freq_real, self.master_freq_imag  # Buffer update needed, since last query was image
        self.last_prompt_text = False
        
        # 1. Calculate the small, dense RoPE table for the actual prompt.
        if self._mrope_interleaved:
            dynamic_freq_real, dynamic_freq_imag, next_pos_id = self.calculate_interleaved(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw
            )
        else:
            dynamic_freq_real, dynamic_freq_imag, next_pos_id = self.calculate(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw
            )
        
        # 2. Get the length of the prompt-specific part.
        prompt_seq_len = dynamic_freq_real.shape[0]

        # 3. Create new, full-sized buffers with the same shape as the master tables.
        hybrid_freq_real = np.empty_like(self.master_freq_real)
        hybrid_freq_imag = np.empty_like(self.master_freq_imag)

        # 4. Overwrite the beginning of the copied tables with the dynamic values.
        hybrid_freq_real[:prompt_seq_len, :] = dynamic_freq_real
        hybrid_freq_imag[:prompt_seq_len, :] = dynamic_freq_imag
        
        # 5. Fill the rest of the table with master table values starting from next_pos_id
        remaining_len = self.max_seq_len - prompt_seq_len
        hybrid_freq_real[prompt_seq_len:, :] = self.master_freq_real[next_pos_id : next_pos_id + remaining_len, :]
        hybrid_freq_imag[prompt_seq_len:, :] = self.master_freq_imag[next_pos_id : next_pos_id + remaining_len, :]

        # 6. Return the new, full-sized, hybrid tables.
        return hybrid_freq_real, hybrid_freq_imag #always do buffer update when image is there