import path from 'node:path';
import { pathToFileURL } from 'node:url';
const SKILL='C:/Users/whats/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const { finalizePresentation }=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({
  explicitTotalSlideCount:10,
  workspaceDir:'C:/Users/whats/Documents/Codex/2026-09-21/wher',
  candidatePath:'C:/Users/whats/Documents/Codex/2026-09-21/wher/presentation_build/candidate.pptx',
  finalPath:'C:/Users/whats/Documents/Codex/2026-09-21/wher/output/ABR_RL_Project_Review_workflow.pptx',
  pythonExecutable:'C:/Users/whats/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-bullet-geometry','--validate-heading-fit'],
  fontPolicy:{basis:'reference',families:['Vidaloka','Montserrat','Arial'],referencePath:'D:/ABR_RL_Project_Review.pptx',referenceSha256:'63792ef732aa037319168688164fa1c49b16e9f649fe3af94168d63ee3e81db8'},
  verifyArtifactToolImport:true,
  receiptPath:'C:/Users/whats/Documents/Codex/2026-09-21/wher/presentation_build/ABR_RL_Project_Review_workflow.validation.json'
});
console.log(JSON.stringify(result,null,2));




