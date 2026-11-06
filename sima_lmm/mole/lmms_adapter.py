"""lmms-eval model interface; upstream owns requests, extraction and scoring."""

from lmms_eval.api.model import lmms

from sima_lmm.mole.multimodal import UnsupportedInput, generation_settings, ordered_content


class ModalixMultimodalLM(lmms):
    def __init__(self, client, record, *, output_budget=None):
        super().__init__()
        self.client = client
        self.record = record
        self.output_budget = output_budget
        self.device = "cpu"
        self.batch_size = 1

    def generate_until(self, requests):
        answers = []
        for request in requests:
            context, kwargs, doc_to_visual, doc_id, task, split = request.args
            row = {"task": task, "split": split, "doc_id": doc_id, "status": "failed"}
            try:
                doc = self.task_dict[task][split][doc_id]
                row["sample_id"] = next((doc[key] for key in
                    ("question_id", "questionId", "id", "image_id") if key in doc), doc_id)
                settings = generation_settings(kwargs, self.output_budget)
                row["generation"] = settings
                content, row["images"] = ordered_content(context, doc_to_visual(doc))
                # Persist the text/ordering, but never duplicate base64 images into answer logs.
                row["content"] = [{"type": part["type"], **(
                    {"text": part["text"]} if part["type"] == "text" else {}
                )} for part in content]
                answer, metadata = self.client.generate([{"role": "user", "content": content}], settings)
                row.update(status="ok", answer=answer, runtime=metadata)
                answers.append(answer)
            except UnsupportedInput as exc:
                row.update(status="unsupported", error=str(exc))
                raise
            except BaseException as exc:
                row["error"] = str(exc) or type(exc).__name__
                raise
            finally:
                self.record(row)
        return answers

    def loglikelihood(self, requests):
        raise UnsupportedInput("Multimodal loglikelihood is unsupported; select a generation task")

    def generate_until_multi_round(self, requests):
        raise UnsupportedInput("Multiround multimodal evaluation is not yet supported")
