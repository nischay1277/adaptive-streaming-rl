import { pathToFileURL } from 'node:url';
import fs from 'node:fs/promises';
const { FileBlob, PresentationFile } = await import(pathToFileURL('C:/Users/whats/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const p=await PresentationFile.importPptx(await FileBlob.load('D:/rl_project/presentation_build/candidate.pptx'));
console.log((await p.inspect({kind:'slide,textbox,image',maxChars:30000})).ndjson);
for(let i=0;i<p.slides.items.length;i++) { const blob=await p.slides.items[i].export({format:'png',scale:1}); await fs.writeFile(`D:/rl_project/presentation_build/final-slide-${String(i+1).padStart(2,'0')}.png`,Buffer.from(await blob.arrayBuffer())); }
