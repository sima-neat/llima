const $ = id => document.getElementById(id);
let health, audio, capture, source, recorder, recording = false, busy = false;
let chunks = [], captured = 0, timer, requestId, abort, nextAudio = 0, playing = [], waveParts = [], waveUrl;
let began = 0, firstPcm = null, playbackStart = null, stopping = false, workletLoaded = false;
const status = text => { $('status').textContent = text; };

function buttons() {
  $('record').disabled = !health || (busy && !recording);
  $('record').textContent = recording ? 'Send recording' : 'Record question';
  $('send').disabled = !health || busy;
  $('stop').disabled = !requestId && playing.length === 0;
}
async function sound() {
  audio ??= new AudioContext();
  await audio.resume();
}
function clearPlayback() {
  stopQueuedPlayback(); waveParts = [];
  $('replay').pause(); $('replay').hidden = true; $('download').hidden = true;
  if (waveUrl) URL.revokeObjectURL(waveUrl);
}
function stopQueuedPlayback() {
  for (const item of playing) { try { item.stop(); } catch {} }
  playing = []; nextAudio = 0;
}
function play(event) {
  const bytes = Uint8Array.from(atob(event.pcm_s16le), c => c.charCodeAt(0));
  waveParts.push(bytes);
  const data = new DataView(bytes.buffer);
  const buffer = audio.createBuffer(1, bytes.length / 2, event.sample_rate);
  const samples = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; ++i) samples[i] = data.getInt16(i * 2, true) / 32768;
  const node = audio.createBufferSource(); node.buffer = buffer; node.connect(audio.destination);
  nextAudio = Math.max(nextAudio, audio.currentTime + 0.05);
  if (firstPcm === null) {
    firstPcm = performance.now() - began;
    playbackStart = firstPcm + (nextAudio - audio.currentTime) * 1000;
  }
  node.start(nextAudio); nextAudio += buffer.duration;
  playing.push(node); buttons();
  node.onended = () => { playing = playing.filter(item => item !== node); buttons(); };
}
function saveWave() {
  const size = waveParts.reduce((n, item) => n + item.length, 0);
  if (!size) return;
  const header = new ArrayBuffer(44), view = new DataView(header);
  function ascii(offset, text) { [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0))); }
  ascii(0, 'RIFF'); view.setUint32(4, 36 + size, true); ascii(8, 'WAVE'); ascii(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 24000, true); view.setUint32(28, 48000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); ascii(36, 'data'); view.setUint32(40, size, true);
  waveUrl = URL.createObjectURL(new Blob([header, ...waveParts], {type: 'audio/wav'}));
  $('replay').src = waveUrl; $('replay').hidden = false;
  $('download').href = waveUrl; $('download').hidden = false;
}
async function request(path, body, contentType) {
  busy = true; requestId = crypto.randomUUID(); abort = new AbortController(); stopping = false;
  buttons(); clearPlayback(); $('answer').textContent = ''; $('metrics').textContent = '';
  status('Generating reply…'); began = performance.now(); firstPcm = playbackStart = null;
  let completed = false;
  try {
    const response = await fetch(`${path}?id=${encodeURIComponent(requestId)}`, {
      method: 'POST', body, headers: {'Content-Type': contentType}, signal: abort.signal,
    });
    if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let pending = '';
    for (;;) {
      const {value, done} = await reader.read();
      pending += decoder.decode(value, {stream: !done});
      let newline;
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline); pending = pending.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === 'text') $('answer').textContent = event.text;
        if (event.type === 'audio' && !stopping) { play(event); status('Playing reply…'); }
        if (event.type === 'error') throw new Error(event.error);
        if (event.type === 'done') {
          completed = true;
          $('metrics').textContent = JSON.stringify({
            server_first_text_ms: event.ttft_visible_ms,
            server_first_pcm_ms: event.first_pcm_ms,
            server_complete_ms: event.request_ms,
            browser_first_pcm_ms: firstPcm,
            browser_playback_scheduled_ms: playbackStart,
            output_audio_seconds: event.output_audio_seconds,
            cancelled: event.cancelled,
          }, null, 2);
          status(event.cancelled ? 'Reply stopped.' : 'Reply complete. You can record another question.');
        }
      }
      if (done) break;
    }
    if (!completed) throw new Error('Reply stream ended before completion');
  } catch (error) {
    status(error.name === 'AbortError' ? 'Reply stopped.' : error.message);
    // A parser/playback failure must also release the model request.
    await fetch(`/api/stop?id=${encodeURIComponent(requestId)}`, {method: 'POST'}).catch(() => {});
    abort.abort();
  } finally {
    saveWave(); requestId = null; abort = null; busy = false; buttons();
  }
}
async function stopRecording() {
  if (!recording) return;
  recording = false; clearTimeout(timer); buttons();
  recorder.disconnect(); source.disconnect(); capture.getTracks().forEach(track => track.stop());
  recorder.port.close();
  try {
    if (!captured) throw new Error('No microphone samples received');
    const raw = new Float32Array(captured); let offset = 0;
    for (const chunk of chunks) { raw.set(chunk, offset); offset += chunk.length; }
    const count = Math.round(raw.length * 16000 / audio.sampleRate);
    const context = new OfflineAudioContext(1, count, 16000);
    const buffer = context.createBuffer(1, raw.length, audio.sampleRate); buffer.copyToChannel(raw, 0);
    const input = context.createBufferSource(); input.buffer = buffer; input.connect(context.destination); input.start();
    const mono = (await context.startRendering()).getChannelData(0);
    const bytes = new ArrayBuffer(mono.length * 4), view = new DataView(bytes);
    mono.forEach((value, i) => view.setFloat32(i * 4, Math.max(-1, Math.min(1, value)), true));
    await request('/api/speech', bytes, 'application/octet-stream');
  } catch (error) { status(error.message); busy = false; buttons(); }
}
$('record').onclick = async () => {
  if (recording) { await stopRecording(); return; }
  busy = true; buttons(); clearPlayback();
  try {
    if (!navigator.mediaDevices || !isSecureContext) throw new Error('Open this page at http://localhost through the SSH tunnel for microphone access.');
    await sound();
    capture = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
    if (!workletLoaded) { await audio.audioWorklet.addModule('/recorder.js'); workletLoaded = true; }
    source = audio.createMediaStreamSource(capture); recorder = new AudioWorkletNode(audio, 'llima-recorder');
    chunks = []; captured = 0; recording = true;
    const maximum = Math.floor(Math.min(2.5, health.max_input_seconds - .01) * audio.sampleRate);
    recorder.port.onmessage = ({data}) => {
      if (!recording) return;
      const part = data.slice(0, maximum - captured); chunks.push(part); captured += part.length;
      if (captured >= maximum) void stopRecording();
    };
    source.connect(recorder); recorder.connect(audio.destination);
    status('Recording… speak a short question, then press Send recording.'); buttons();
    timer = setTimeout(() => void stopRecording(), 2600);
  } catch (error) {
    capture?.getTracks().forEach(track => track.stop());
    status(error.message); busy = false; buttons();
  }
};
$('text-form').onsubmit = async event => {
  event.preventDefault(); if (busy || !$('text').value.trim()) return;
  busy = true; buttons();
  try { await sound(); await request('/api/text', JSON.stringify({text: $('text').value}), 'application/json'); }
  catch (error) { status(error.message); busy = false; buttons(); }
};
$('replay').onplay = () => { stopQueuedPlayback(); buttons(); };
$('stop').onclick = async () => {
  if (!requestId) { stopQueuedPlayback(); status('Playback stopped.'); buttons(); return; }
  stopping = true;
  clearPlayback(); status('Stopping reply…');
  await fetch(`/api/stop?id=${encodeURIComponent(requestId)}`, {method: 'POST'}).catch(() => {});
};
fetch('/api/health').then(response => response.json()).then(value => {
  health = value;
  $('limit').textContent = `Microphone limit: ${Math.min(2.5, health.max_input_seconds - .01).toFixed(2)} seconds per question. Use your computer’s microphone and speakers or headphones.`;
  status(health.busy ? 'Model is serving another request.' : 'Ready.'); buttons();
}).catch(error => status(`Server unavailable: ${error.message}`));
