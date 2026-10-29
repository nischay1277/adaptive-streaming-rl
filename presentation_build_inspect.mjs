import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const runtime = "C:/Users/whats/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { FileBlob, PresentationFile } = await import(pathToFileURL(runtime).href);
const input = "D:/ABR_RL_Project_Review.pptx";
const out = "D:/rl_project/presentation_build";
await fs.mkdir(out, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(input));
const inspection = await presentation.inspect({
  select: "$.slides[*]",
  include: ["id", "index", "layout", "title", "elements.id", "elements.kind", "elements.text", "elements.alt", "elements.title", "notes.text"],
});
await fs.writeFile(path.join(out, "source.ndjson"), inspection.ndjson, "utf8");
console.log(`slides=${presentation.slides.items.length}`);
console.log(inspection.ndjson);
for (const [i, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(out, `source-slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
const layout = await presentation.inspect({
  select: "$.slides[*]",
  include: ["id", "index", "width", "height", "elements.id", "elements.kind", "elements.text", "elements.position", "elements.style"],
});
await fs.writeFile(path.join(out, "source-layout.ndjson"), layout.ndjson, "utf8");
