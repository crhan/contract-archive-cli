"""
ocr-cli 主命令行。

用法（建议结合 uv）：
  uv run ocr-cli run --pipeline dashscope ./input/contract.pdf
  uv run ocr-cli run --pipeline all       ./input/contract.pdf
  uv run ocr-cli extract ./output/dashscope          # 基于该路 OCR 结果做合同字段抽取
  uv run ocr-cli compare ./output                    # 横向对比三路输出
  uv run ocr-cli serve --port 8000                   # 启 FastAPI 服务
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table as RichTable

app = typer.Typer(help="Document Intelligence Playground CLI")
console = Console()
load_dotenv()  # 自动加载项目根目录 .env

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


PIPELINES = ("dashscope", "paddleocr", "mineru")


@app.command()
def run(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="待处理 PDF"),
    pipeline: str = typer.Option(
        "all", "--pipeline", "-p", help="dashscope / paddleocr / mineru / all"
    ),
    out_root: Path = typer.Option(
        Path(os.getenv("OUTPUT_DIR", "./output")), "--out", "-o"
    ),
    device: Optional[str] = typer.Option(None, help="auto / mps / cuda / cpu"),
) -> None:
    """跑指定 OCR pipeline，输出统一 schema。"""
    from .pipelines import get_pipeline

    targets = PIPELINES if pipeline == "all" else (pipeline,)
    failures = 0
    for name in targets:
        if name not in PIPELINES:
            console.print(f"[red]unknown pipeline {name}[/red]")
            raise typer.Exit(2)
        out_dir = out_root / name
        console.rule(f"[bold cyan]{name}[/bold cyan]")
        try:
            pl = get_pipeline(name, device=device)
            result = pl.run(pdf, out_dir)
            console.print(
                f"[green]✓[/green] {name}: {result.meta.duration_seconds:.2f}s, "
                f"{len(result.layout)} layout blocks, "
                f"{result.structured.pages} pages → {out_dir}"
            )
        except Exception as e:
            console.print(f"[red]✗[/red] {name} failed: {e}")
            logging.getLogger(__name__).exception("%s failed", name)
            failures += 1
    if failures:
        raise typer.Exit(1)


@app.command()
def extract(
    ocr_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    no_llm: bool = typer.Option(False, "--no-llm", help="只用 rule，跳过 LLM 调用"),
) -> None:
    """基于某路 OCR 输出做合同语义抽取。"""
    from .extraction import extract_contract
    from .schemas import FILE_EXTRACTION, FILE_EXTRACTION_CONF, FILE_MARKDOWN, FILE_RAW_TEXT

    md_path = ocr_dir / FILE_MARKDOWN
    txt_path = ocr_dir / FILE_RAW_TEXT
    if not md_path.exists() and not txt_path.exists():
        console.print(f"[red]neither {md_path.name} nor {txt_path.name} found[/red]")
        raise typer.Exit(2)
    # 优先用 raw_text：它是 pipeline 已清洗过的版本（去掉了 LaTeX/转义符等噪声）
    # markdown.md 仅在 raw_text 缺失时兜底
    document_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else md_path.read_text(encoding="utf-8")

    console.print(f"[cyan]extracting from {ocr_dir} (chars={len(document_text)})[/cyan]")
    extraction, conf = extract_contract(document_text, llm_enabled=not no_llm)

    (ocr_dir / FILE_EXTRACTION).write_text(
        extraction.model_dump_json(indent=2), encoding="utf-8"
    )
    (ocr_dir / FILE_EXTRACTION_CONF).write_text(
        conf.model_dump_json(indent=2), encoding="utf-8"
    )

    table = RichTable(title=f"Extraction · {ocr_dir.name}")
    table.add_column("field", style="cyan")
    table.add_column("value", overflow="fold")
    table.add_column("conf", style="magenta")
    table.add_column("source", style="green")
    table.add_row("contract_name", str(extraction.contract_name), f"{conf.contract_name.confidence:.2f}", conf.contract_name.value_source)
    table.add_row("party_a", str(extraction.party_a), f"{conf.party_a.confidence:.2f}", conf.party_a.value_source)
    table.add_row("party_b", str(extraction.party_b), f"{conf.party_b.confidence:.2f}", conf.party_b.value_source)
    table.add_row("amount", str(extraction.amount), f"{conf.amount.confidence:.2f}", conf.amount.value_source)
    table.add_row("sign_date", str(extraction.sign_date), f"{conf.sign_date.confidence:.2f}", conf.sign_date.value_source)
    table.add_row("expire_date", str(extraction.expire_date), f"{conf.expire_date.confidence:.2f}", conf.expire_date.value_source)
    table.add_row("auto_renewal", str(extraction.auto_renewal), f"{conf.auto_renewal.confidence:.2f}", conf.auto_renewal.value_source)
    table.add_row("risk_clauses", f"{len(extraction.risk_clauses)} items", f"{conf.risk_clauses.confidence:.2f}", conf.risk_clauses.value_source)
    table.add_row("[bold]overall[/bold]", "-", f"[bold]{conf.overall:.2f}[/bold]", "-")
    console.print(table)


@app.command()
def compare(
    out_root: Path = typer.Argument(Path("./output"), exists=True, file_okay=False),
    report_path: Optional[Path] = typer.Option(
        None, "--report", help="对比报告输出路径（默认 out_root/compare_report.md）"
    ),
) -> None:
    """横向对比三路 OCR + extraction 结果。"""
    from .compare import generate_report

    report = generate_report(out_root)
    out_path = report_path or (out_root / "compare_report.md")
    out_path.write_text(report, encoding="utf-8")
    console.print(f"[green]✓ compare report → {out_path}[/green]")
    console.print(report)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """启动 FastAPI 服务。"""
    import uvicorn

    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)


@app.command()
def benchmark(
    pdf: Path = typer.Argument(..., exists=True),
    out_root: Path = typer.Option(Path("./output")),
    rounds: int = typer.Option(1, "--rounds", "-r", help="每路重复次数"),
) -> None:
    """基准测试：跑三路并采集时间/字符数/layout 数。"""
    from .compare import benchmark_pipelines

    results = benchmark_pipelines(pdf, out_root, rounds=rounds)
    table = RichTable(title="Benchmark")
    table.add_column("pipeline")
    table.add_column("duration_s", justify="right")
    table.add_column("raw_chars", justify="right")
    table.add_column("md_chars", justify="right")
    table.add_column("layout_blocks", justify="right")
    table.add_column("tables", justify="right")
    table.add_column("status")
    for r in results:
        table.add_row(
            r["pipeline"],
            f"{r['duration_s']:.2f}",
            str(r["raw_chars"]),
            str(r["md_chars"]),
            str(r["layout_blocks"]),
            str(r["tables"]),
            r["status"],
        )
    console.print(table)
    json_path = out_root / "benchmark.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✓ benchmark.json → {json_path}[/green]")


if __name__ == "__main__":
    app()
