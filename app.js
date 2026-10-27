const $=id=>document.getElementById(id);let data=[],detector,stream,running=false,recognizing=false,collect=null,frame,last=0,candidate='',streak=0,latestHand=null,latestTime=0,pendingPose=null,letterToAdd='';try{data=JSON.parse(localStorage.getItem('signflow-v1')||'[]')}catch{}if(!valid(data))data=[];
function valid(d){return Array.isArray(d)&&d.length<=30&&d.every(c=>typeof c.name==='string'&&c.name.length>0&&c.name.length<=40&&Array.isArray(c.samples)&&c.samples.length<=2000&&c.samples.every(s=>Array.isArray(s)&&s.length===63&&s.every(n=>typeof n==='number'&&Number.isFinite(n))))}
function save(){try{localStorage.setItem('signflow-v1',JSON.stringify(data));render();return true}catch{$('progress').textContent='Não foi possível salvar neste navegador. Exporte os dados antes de fechar.';render();return false}}
function render(){const old=$('classes').value;$('classes').replaceChildren(...data.map((c,i)=>new Option(`${c.name} · ${c.samples.length} captura(s)`,i)));if(data[old])$('classes').value=old;$('count').textContent=data.filter(c=>c.samples.length).length;$('recognition').disabled=!running||!data.some(c=>c.samples.length);$('capture').disabled=!running;$('remove').disabled=!data.length;$('export').disabled=!data.length;if(!recognizing)$('detail').textContent=data.some(c=>c.samples.length)?'Sinais salvos. Ative a câmera e inicie o reconhecimento.':'Salve seu primeiro sinal para começar.'}
function resetResult(message='Aguardando um sinal'){stableCandidateSince=0;lastPredictionAt=0;showHold(0,'Aguardando um sinal estável.');offerLetter('');$('word').textContent=message;$('score').textContent='—';$('bar').style.width='0';candidate='';streak=0}
$('help').onclick=()=>$('guide').showModal();
$('add').onsubmit=e=>{e.preventDefault();const name=$('name').value.trim();if(!name||!pendingPose)return;const existing=data.find(c=>c.name.toLowerCase()===name.toLowerCase());if(existing){if(existing.samples.length>=2000)return alert('Limite de capturas atingido.');existing.samples.push(pendingPose)}else{if(data.length>=30)return alert('Limite de 30 sinais.');data.push({name,samples:[pendingPose]})}const persisted=save();$('classes').value=data.findIndex(c=>c.name.toLowerCase()===name.toLowerCase());pendingPose=null;$('captured').hidden=true;$('name').value='';$('progress').textContent=persisted?`Sinal “${name}” salvo. Capture o próximo ou teste em Reconhecer.`:'Sinal disponível nesta sessão, mas não foi salvo no navegador. Exporte seus dados.'};
$('discard').onclick=()=>{pendingPose=null;$('captured').hidden=true;$('progress').textContent='Captura descartada. Mostre uma pose e capture novamente.'};
$('start').onclick=async()=>{const b=$('start');b.disabled=true;b.textContent='Aguardando permissão…';$('status').textContent='Permita o acesso à câmera no aviso do navegador. Se abriu dentro do ChatGPT, tente abrir o link em uma nova aba.';try{if(!navigator.mediaDevices?.getUserMedia)throw Error('Câmera indisponível neste navegador. Use HTTPS e um navegador atualizado.');stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'},audio:false});$('video').srcObject=stream;await $('video').play();$('placeholder').hidden=true;$('stop').disabled=false;$('status').textContent='Câmera conectada. Carregando detector de mãos…';if(!detector){const {FilesetResolver,HandLandmarker}=await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs');const vision=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm');detector=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'},runningMode:'VIDEO',numHands:2})}running=true;$('placeholder').hidden=true;$('videoLabel').hidden=false;$('cameraState').textContent='● Ativa';$('stop').disabled=false;$('progress').textContent='Mostre uma mão e clique em Capturar sinal.';render();frame=requestAnimationFrame(loop)}catch(e){stop();$('status').textContent=e.name==='NotAllowedError'?'Permissão negada. Libere a câmera nas configurações do navegador.':e.name==='NotFoundError'?'Nenhuma câmera encontrada. Conecte uma webcam.':'Não foi possível iniciar: '+e.message}finally{b.disabled=false;b.textContent='Ativar câmera'}};
function stop(){endAttempt();latestHand=null;running=false;recognizing=false;if(collect){save();collect=null;$('progress').textContent='Coleta interrompida. Exemplos já capturados foram salvos.'}cancelAnimationFrame(frame);stream?.getTracks().forEach(t=>t.stop());$('video').srcObject=null;$('placeholder').hidden=false;$('videoLabel').hidden=true;$('cameraState').textContent='● Desligada';$('stop').disabled=true;$('recognition').textContent='Iniciar reconhecimento';$('status').textContent='Sua câmera está desligada.';resetResult();render()}
$('stop').onclick=stop;window.addEventListener('pagehide',stop);
function features(hand){const o=hand[0],scale=Math.max(.001,Math.hypot(hand[9].x-o.x,hand[9].y-o.y,hand[9].z-o.z));return hand.flatMap(p=>[(p.x-o.x)/scale,(p.y-o.y)/scale,(p.z-o.z)/scale])}
// High-tolerance experimental matching; existing 63-coordinate samples stay compatible.
function directPoseDistance(a,b,relaxed=true){
 let best=Infinity;
 for(const degrees of (relaxed?[-25,-12,0,12,25]:[0])){
  const angle=degrees*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);let sum=0;
  for(let i=0;i<63;i+=3){
   const x=b[i]*c-b[i+1]*s,y=b[i]*s+b[i+1]*c;
   sum+=(a[i]-x)**2+(a[i+1]-y)**2+(relaxed?.35:1)*(a[i+2]-b[i+2])**2;
  }
  best=Math.min(best,Math.sqrt(sum/63));
 }
 return best;
}
function poseDistance(a,b,relaxed=true){
 const direct=directPoseDistance(a,b,relaxed);
 if(!relaxed)return direct;
 const mirrored=b.map((v,i)=>i%3===0?-v:v);
 return Math.min(direct,directPoseDistance(a,mirrored,true));
}
function classify(f){
 if(!Array.isArray(f)||f.length!==63||!f.every(Number.isFinite))return null;
 const ranks=data.filter(c=>c.samples.length).map(c=>({
  name:c.name,command:false,d:c.samples.reduce((best,s)=>Math.min(best,poseDistance(f,s)),Infinity)
 })).sort((a,b)=>a.d-b.d);
 // Confirmation remains strict because it inserts text without a click.
 const commandDistance=commandSamples.reduce((best,s)=>Math.min(best,poseDistance(f,s,false)),Infinity);
 const strictLetterDistance=data.reduce((best,c)=>c.samples.reduce((d,s)=>Math.min(d,poseDistance(f,s,false)),best),Infinity);
 if(commandDistance<=.18&&strictLetterDistance-commandDistance>=.035)
  return {name:'Gesto de confirmação',command:true,score:Math.max(0,1-commandDistance/.5)};
 if(!ranks.length||ranks[0].d>.55)return null;
 return {name:ranks[0].name,command:false,score:Math.max(0,1-ranks[0].d/.65)};
}

const edges=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
function loop(t){if(!running)return;try{if(t-last>90&&$('video').readyState>=2){last=t;const result=detector.detectForVideo($('video'),t),hands=result.landmarks;latestHand=hands.length===1?hands[0]:null;latestTime=t;const canvas=$('overlay');canvas.width=$('video').videoWidth;canvas.height=$('video').videoHeight;const ctx=canvas.getContext('2d');ctx.strokeStyle='#69e7ab';ctx.lineWidth=2;ctx.fillStyle='#dcffe9';for(const h of hands){for(const [a,b]of edges){ctx.beginPath();ctx.moveTo(h[a].x*canvas.width,h[a].y*canvas.height);ctx.lineTo(h[b].x*canvas.width,h[b].y*canvas.height);ctx.stroke()}for(const p of h){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,3,0,Math.PI*2);ctx.fill()}}$('status').textContent=hands.length===1?'Mão detectada · Mantenha a pose estável':hands.length>1?'Use apenas uma mão nesta versão.':'Posicione uma mão na área de captura.';if(recognizing)processPrediction(hands.length===1?classify(features(hands[0])):null);}}catch(e){stop();$('status').textContent='Falha no processamento. Tente ativar a câmera novamente.';return}frame=requestAnimationFrame(loop)}
$('capture').onclick=()=>{if(!running||!latestHand||performance.now()-latestTime>750){$('progress').textContent='Nenhuma pose disponível. Mostre apenas uma mão inteira à câmera e tente novamente.';return}if(pendingPose&&!confirm('Substituir a captura ainda não salva?'))return;recognizing=false;$('recognition').textContent='Iniciar reconhecimento';pendingPose=features(latestHand);const canvas=$('posePreview'),ctx=canvas.getContext('2d'),video=$('video');canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();$('captured').hidden=false;$('progress').textContent='Pose capturada! Agora indique o valor e salve.';$('name').focus()};
$('recognition').onclick=()=>{if(recognizing){endAttempt();resetResult();$('detail').textContent='Reconhecimento pausado.';return}startAttempt()};
$('remove').onclick=()=>{const i=Number($('classes').value);if(data[i]&&confirm(`Excluir ${data[i].name} e todos os exemplos?`)){data.splice(i,1);save();resetResult()}};
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({version:1,classes:data},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='alem-da-voz-sinais.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
$('import').onclick=()=>$('file').click();$('file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>20000000)throw Error();const x=JSON.parse(await file.text());if(x.version!==1||!valid(x.classes))throw Error();if(data.length&&!confirm('Substituir os exemplos atuais pelo arquivo?'))return;collect=null;recognizing=false;pendingPose=null;$('captured').hidden=true;data=x.classes;save();resetResult();$('progress').textContent='Dados importados com sucesso.'}catch{alert('Arquivo inválido. Importe um JSON exportado pelo Além da Voz.')}finally{e.target.value=''}};render();

$('status').textContent='Pronto. Clique em Ativar câmera para começar.';

function offerLetter(value){letterToAdd=value;$('appendLetter').disabled=!value;$('appendLetter').textContent=value?`Adicionar “${value}” ao texto`:'Adicionar letra reconhecida'}
function persistText(message){try{localStorage.setItem('signflow-text-v1',$('composition').value);$('textStatus').textContent=message||'Texto salvo neste navegador.'}catch{$('textStatus').textContent='Não foi possível salvar. Copie o texto antes de fechar.'}}
try{$('composition').value=localStorage.getItem('signflow-text-v1')||''}catch{}
$('appendLetter').onclick=()=>submitLetter();
$('composition').oninput=()=>persistText();
$('space').onclick=()=>{$('composition').value+=' ';persistText('Espaço adicionado.')};
$('backspace').onclick=()=>{const letters=Array.from($('composition').value);letters.pop();$('composition').value=letters.join('');persistText('Último caractere apagado.')};
$('clearText').onclick=()=>{if(!$('composition').value||!confirm('Limpar todo o texto formado?'))return;$('composition').value='';persistText('Texto limpo. Você pode começar novamente.')};
$('copyText').onclick=async()=>{try{await navigator.clipboard.writeText($('composition').value);$('textStatus').textContent='Texto copiado!'}catch{$('composition').focus();$('composition').select();$('textStatus').textContent='Selecionei o texto. Use Ctrl+C para copiar.'}};

let attemptTimer=null;
function endAttempt(){clearTimeout(attemptTimer);recognizing=false;$('recognition').textContent='Iniciar reconhecimento';$('cameraCard').classList.remove('searching');}
function startAttempt(){endAttempt();resetResult();$('cameraCard').classList.remove('timeout','recognized');$('recognize').classList.remove('timeout','recognized');recognizing=true;$('cameraCard').classList.add('searching');$('recognition').textContent='Pausar reconhecimento';$('detail').textContent='Versão espelhamento · Alta tolerância ativa. Mostre o sinal e confira a letra.';attemptTimer=setTimeout(()=>{if(recognizing&&!letterToAdd){$('detail').textContent='Continuo procurando. Ajuste a mão; não é preciso reiniciar.';}},15000);}
function recognitionSuccess(){clearTimeout(attemptTimer);$('cameraCard').classList.remove('searching');$('cameraCard').classList.add('recognized');$('recognize').classList.add('recognized');$('recognition').textContent='Pausar reconhecimento';}
const cameraHome=document.querySelector('.workspace');
$('settingsOpen').onclick=()=>{endAttempt();$('settings').showModal();$('cameraDock').append($('cameraCard'));if(stream)$('video').play().catch(()=>{});selectPanel('signals');};
$('settingsClose').onclick=()=>$('settings').close();
$('settings').addEventListener('close',()=>{cameraHome.prepend($('cameraCard'));if(stream)$('video').play().catch(()=>{});$('settingsOpen').focus();});
function selectPanel(panel){document.querySelectorAll('[data-panel]').forEach(b=>{if(b.dataset.panel===panel)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});$('signals').hidden=panel!=='signals';$('appearance').hidden=panel!=='appearance';$('confirmation').hidden=panel!=='confirmation';if(panel==='signals'||panel==='confirmation'){ $(panel==='signals'?'cameraDock':'commandCameraDock').append($('cameraCard'));if(stream)$('video').play().catch(()=>{});}}
document.querySelectorAll('[data-panel]').forEach(b=>b.onclick=()=>selectPanel(b.dataset.panel));
const palettes={roxo:['Roxo','#69508f','#dec7ff'],rosa:['Rosa','#92426c','#ffb1d4'],azul:['Azul','#315f9d','#adcaff'],verde:['Verde','#28694e','#9ad7b6'],terracota:['Terracota','#925035','#ffb599']};
let theme={color:'roxo',mode:'light'};try{const saved=JSON.parse(localStorage.getItem('alem-theme'));if(saved&&palettes[saved.color]&&['light','dark'].includes(saved.mode))theme=saved;}catch{}
Object.entries(palettes).forEach(([key,p])=>{const b=document.createElement('button');b.textContent=p[0];b.dataset.color=key;b.style.setProperty('--swatch',p[1]);b.onclick=()=>{theme.color=key;applyTheme(true)};$('colors').append(b)});
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{theme.mode=b.dataset.mode;applyTheme(true)});
function applyTheme(persist=false){document.documentElement.dataset.mode=theme.mode;document.documentElement.style.setProperty('--primary',palettes[theme.color][theme.mode==='dark'?2:1]);document.querySelectorAll('[data-color]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.color===theme.color)));document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===theme.mode)));if(persist)try{localStorage.setItem('alem-theme',JSON.stringify(theme));$('themeStatus').textContent='Aparência salva.'}catch{$('themeStatus').textContent='Aparência aplicada apenas nesta sessão.'}}
applyTheme();

let commandSamples=[],pendingCommand=null,releaseKey='',releaseFrames=0;
try{const saved=JSON.parse(localStorage.getItem('alem-confirm-gesture')||'[]');if(valid([{name:'command',samples:saved}]))commandSamples=saved;}catch{}
function updateCommandStatus(message){$('commandStatus').textContent=message||(commandSamples.length?'Gesto de confirmação ativo. Faça uma letra e depois este gesto para adicionar.':'Nenhum gesto configurado. Capture um gesto para ativar.');$('removeCommand').disabled=!commandSamples.length;}
function predictionKey(p){return p?(p.command?'command':'letter:'+p.name):'';}
let continuousRecognition=true,missingPredictionFrames=0;
try{continuousRecognition=localStorage.getItem('alem-continuous')!=='false'}catch{}
let stableCandidateSince=0,lastPredictionAt=0,noPredictionSince=null;
const LETTER_HOLD_MS=1100,CONFIRM_HOLD_MS=800;
function showHold(percent,message){
 const meter=$('recognitionHold');if(meter)meter.value=percent;
 const caption=$('recognitionHoldLabel');if(caption)caption.textContent=message;
}
function processPrediction(p){
 const now=performance.now(),key=predictionKey(p);
 if(releaseKey){
  stableCandidateSince=0;candidate='';
  if(key===releaseKey){releaseFrames=0;showHold(0,'Solte o gesto para continuar.');return}
  if(++releaseFrames<3)return;
  releaseKey='';releaseFrames=0;
 }
 if(!p){
  candidate='';streak=0;stableCandidateSince=0;lastPredictionAt=0;
  if(noPredictionSince===null)noPredictionSince=now;
  showHold(0,letterToAdd?'Letra pronta. Faça o gesto de confirmação.':'Aguardando um sinal estável.');
  if(continuousRecognition&&now-noPredictionSince>=1800)resetResult();
  if(!letterToAdd){$('word').textContent='Aguardando um sinal';$('detail').textContent=latestHand?'Mão detectada. Mantenha uma pose cadastrada.':'Mostre uma mão inteira à câmera.';}
  return;
 }
 noPredictionSince=null;missingPredictionFrames=0;
 if(letterToAdd&&!p.command&&!continuousRecognition)return;
 if(p.command&&!letterToAdd){candidate='';stableCandidateSince=0;showHold(0,'Reconheça uma letra antes de confirmar.');return}
 if(!p.command&&letterToAdd===p.name){
  candidate=key;stableCandidateSince=now;lastPredictionAt=now;
  showHold(100,'Letra pronta. Faça o gesto de confirmação ou mude o sinal.');
  return;
 }
 if(candidate!==key||!lastPredictionAt||now-lastPredictionAt>450){
  candidate=key;stableCandidateSince=now;streak=0;
 }
 lastPredictionAt=now;streak++;
 const duration=p.command?CONFIRM_HOLD_MS:LETTER_HOLD_MS;
 const elapsed=now-stableCandidateSince,percent=Math.min(100,Math.floor(elapsed/duration*100));
 showHold(percent,p.command?'Confirmando a letra… '+percent+'%':'Estabilizando o sinal… '+percent+'%');
 if(!letterToAdd)$('word').textContent='Analisando o sinal…';
 $('detail').textContent=p.command?'Mantenha o gesto de confirmação até completar.':'Mantenha o mesmo sinal por cerca de 1 segundo.';
 if(elapsed<duration||streak<3)return;
 if(p.command){submitLetter('command');return}
 recognitionSuccess();offerLetter(p.name);$('word').textContent=p.name;
 $('detail').textContent=commandSamples.length?'Letra pronta. Faça o gesto de confirmação para salvar.':'Letra pronta. Clique em Adicionar para salvar.';
 $('score').textContent=p.score.toFixed(2);$('bar').style.width=p.score*100+'%';
 showHold(100,'Sinal estabilizado.');
}
function submitLetter(source){if(!letterToAdd)return;const value=letterToAdd;releaseKey=source==='command'?'command':'letter:'+value;releaseFrames=0;$('composition').value+=value;offerLetter('');persistText(`“${value}” adicionado. Faça o próximo sinal.`);if(running)startAttempt();}
$('captureCommand').onclick=()=>{if(!running||!latestHand||performance.now()-latestTime>750){updateCommandStatus('Ative a câmera e mostre uma mão inteira para capturar.');return;}pendingCommand=features(latestHand);const canvas=$('commandPreview'),ctx=canvas.getContext('2d'),video=$('video');canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();$('commandCaptured').hidden=false;updateCommandStatus('Confira a captura e salve para usar como confirmação.');};
$('saveCommand').onclick=()=>{if(!pendingCommand)return;const previous=commandSamples;commandSamples=[pendingCommand];try{localStorage.setItem('alem-confirm-gesture',JSON.stringify(commandSamples));pendingCommand=null;$('commandCaptured').hidden=true;updateCommandStatus('Gesto salvo! Faça uma letra e depois este gesto para adicionar.');}catch{commandSamples=previous;updateCommandStatus('Não foi possível salvar. Tente novamente.');}};
$('removeCommand').onclick=()=>{try{localStorage.removeItem('alem-confirm-gesture');commandSamples=[];pendingCommand=null;$('commandCaptured').hidden=true;updateCommandStatus('Gesto desativado. Você pode continuar usando o botão Adicionar.');}catch{updateCommandStatus('Não foi possível desativar o gesto.');}};
updateCommandStatus();



const continuousOption=document.createElement('label');
continuousOption.style.cssText='display:flex;align-items:center;gap:10px;margin:16px 0;font-size:13px;cursor:pointer';
const continuousCheckbox=document.createElement('input');
continuousCheckbox.type='checkbox';continuousCheckbox.id='continuousRecognition';
continuousCheckbox.checked=continuousRecognition;
continuousCheckbox.style.cssText='width:18px;height:18px;margin:0;flex:none';
const continuousCaption=document.createElement('span');
continuousCaption.textContent='Reconhecimento contínuo';
continuousOption.append(continuousCheckbox,continuousCaption);
$('recognition').insertAdjacentElement('afterend',continuousOption);
continuousCheckbox.addEventListener('change',()=>{
 continuousRecognition=continuousCheckbox.checked;missingPredictionFrames=0;releaseKey='';releaseFrames=0;resetResult();
 try{localStorage.setItem('alem-continuous',String(continuousRecognition))}catch{}
 $('detail').textContent=continuousRecognition?'Modo contínuo ativado. Inicie o reconhecimento e mude os sinais livremente.':'Modo individual ativado. Cada letra fica aguardando confirmação.';
});


const holdBox=document.createElement('div');
holdBox.style.cssText='margin:16px 0;text-align:left';
const holdLabel=document.createElement('div');
holdLabel.id='recognitionHoldLabel';holdLabel.textContent='Mantenha a letra por cerca de 1 segundo.';
holdLabel.style.cssText='font-size:12px;margin-bottom:8px';
const holdMeter=document.createElement('progress');holdMeter.id='recognitionHold';
holdMeter.max=100;holdMeter.value=0;holdMeter.setAttribute('aria-label','Estabilização do sinal');
holdMeter.style.cssText='width:100%;height:8px;accent-color:var(--primary,#69508f)';
holdBox.append(holdLabel,holdMeter);$('recognition').insertAdjacentElement('beforebegin',holdBox);

