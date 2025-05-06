from __future__ import annotations

import os, argparse, time, textwrap
from pathlib import Path
import tenacity
from datetime import datetime
from dotenv import load_dotenv
import google.genai as genai           # using google-genai SDK
import arxiv

# Load API keys
load_dotenv()
_API_KEY = os.getenv("GOOGLE_API_KEY")
_S2_KEY  = os.getenv("S2_API_KEY")
if not _API_KEY:
    raise EnvironmentError("環境変数 GOOGLE_API_KEY が設定されていません。")

_CLIENT = genai.Client(api_key=_API_KEY)
DEFAULT_MODEL = "gemini-2.0-flash"
RESEARCH_DIR_PATH = "MATH_research_dir"

@tenacity.retry(stop=tenacity.stop_after_attempt(5),
                wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
                reraise=True)
def query_model(prompt: str, model_str: str = DEFAULT_MODEL) -> str:
    resp = _CLIENT.models.generate_content(
        model=model_str,
        contents=prompt,
    )
    return resp.text.strip()

class ArxivSearch:
    def __init__(self, delay: float = 1.0):
        self.client = arxiv.Client(delay_seconds=delay, page_size=10)

    def search(self, query: str, n: int):
        q = query.replace("_", " ")
        return list(self.client.results(arxiv.Search(query=q, max_results=n)))

class SemanticScholarSearch:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: str):
        self.key = api_key

    def search(self, query: str, n: int):
        import requests
        headers = {"x-api-key": self.key}
        params = {"query": query, "limit": n, "fields": "title,authors,year,abstract,venue,journal,citationCount,fieldsOfStudy,openAccessPdf,url"}
        resp = requests.get(self.BASE_URL, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])

class LaboratoryWorkflow:
    def __init__(self, *, topic: str, n_papers: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lab_dir = Path(RESEARCH_DIR_PATH) / f"run_{ts}"
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        self.topic = topic
        self.n_papers = n_papers
        self.searcher_arxiv = ArxivSearch()
        self.searcher_s2 = SemanticScholarSearch(_S2_KEY) if _S2_KEY else None
        self.lit_summary = ""
        self.plan_text = ""
        self.additional_query = ""
        self.additional_summary = ""
        self.exp_results = ""
        self.discussion = ""
        self.conclusion = ""

    def literature_review(self):
        print(f"[ステップ1] 文献レビュー: '{self.topic}' を arXiv と Semantic Scholar から検索中…")
        bullets = []
        for i, p in enumerate(self.searcher_arxiv.search(self.topic, self.n_papers), 1):
            prompt = textwrap.dedent(
                f"""
                以下の論文のタイトルと要旨（abstract）に基づき、貢献、手法、結果を日本語で3文以内に要約してください。
                論文タイトル: {p.title}
                要旨: {p.summary}
                """
            )
            bullets.append(f"- **(arXiv) {p.title}** — {query_model(prompt)}")
        if self.searcher_s2:
            for i, p in enumerate(self.searcher_s2.search(self.topic, self.n_papers), 1):
                title = p.get("title")
                abstract = p.get("abstract", "")
                prompt = textwrap.dedent(
                    f"""
                    以下の論文のタイトルと要旨（abstract）に基づき、主な発見を日本語で2文以内に要約してください。
                    論文タイトル: {title}
                    要旨: {abstract}
                    """
                )
                bullets.append(f"- **(S2) {title}** — {query_model(prompt)}")
        md = "### 文献レビュー\n" + "\n".join(bullets)
        (self.lab_dir / "literature_review.md").write_text(md)
        self.lit_summary = md
        print(f"文献レビュー結果を保存 → {self.lab_dir / 'literature_review.md'}")

    def plan_formulation(self):
        print("[ステップ2] 実験計画の立案…")
        system_prompt = textwrap.dedent(
            f"""
            あなたは博士課程の学生を指導するポスドクです。以下の文献レビューを踏まえて、
            『{self.topic}』をさらに調査するための異なる2つの実験計画を提案してください。
            各実験計画について、目的、方法、必要なデータ、期待される結果を番号付きMarkdownリストで記述してください。

            文献レビュー内容:
            {self.lit_summary}
            """
        )
        self.plan_text = query_model(system_prompt)
        (self.lab_dir / "plan.md").write_text(self.plan_text)
        print(f"実験計画を保存 → {self.lab_dir / 'plan.md'}")

    def additional_literature_review(self):
        print("[ステップ3] 追加文献レビュー: 実験計画に基づく結果・考察の文献レビュー…")
        prompt_query = textwrap.dedent(
            f"""
            以下の実験計画に基づき、新たに必要な論文を調べるための検索クエリを3単語（スペース区切り）で生成してください。
            実験計画:
            {self.plan_text}
            """
        )
        self.additional_query = query_model(prompt_query)
        print(f"追加検索クエリ: {self.additional_query}")
        bullets = []
        for p in self.searcher_arxiv.search(self.additional_query, self.n_papers):
            prompt = textwrap.dedent(
                f"""
                以下の論文のタイトルと要旨（abstract）に基づき、結果および考察を中心に日本語で3文以内に要約してください。
                論文タイトル: {p.title}
                要旨: {p.summary}
                """
            )
            bullets.append(f"- **(arXiv) {p.title}** — {query_model(prompt)}")
        if self.searcher_s2:
            for p in self.searcher_s2.search(self.additional_query, self.n_papers):
                title = p.get("title")
                abstract = p.get("abstract", "")
                prompt = textwrap.dedent(
                    f"""
                    以下の論文のタイトルと要旨（abstract）に基づき、結果および考察を中心に日本語で3文以内に要約してください。
                    論文タイトル: {title}
                    要旨: {abstract}
                    """
                )
                bullets.append(f"- **(S2) {title}** — {query_model(prompt)}")
        md = "### 追加文献レビュー: 結果・考察\n" + "\n".join(bullets)
        (self.lab_dir / "additional_review.md").write_text(md)
        self.additional_summary = md
        print(f"追加文献レビュー結果を保存 → {self.lab_dir / 'additional_review.md'}")

    def data_preparation(self):
        print("[ステップ4] データ準備: 追加文献レビュー結果を実験結果として生成…")
        prompt_results = textwrap.dedent(
            f"""
            {self.additional_summary}を元に
            {self.plan_text}に即した実験結果をMECEにまとめてください。
            """
        )
        self.exp_results = query_model(prompt_results)
        print(f"生成された実験結果: {self.exp_results}")

    def generate_discussion_and_conclusion(self):
        prompt_disc = textwrap.dedent(
            f"""
            あなたはNovel賞を取るような科学者の一人です。平易な言葉に甘えず、
            以下の実験結果を踏まえて、考察を日本語でMECEに生成してください。
            実験結果: {self.exp_results}
            """
        )
        self.discussion = query_model(prompt_disc)
        prompt_conc = textwrap.dedent(
            f"""
            以下の考察を踏まえて、結論を日本語でMECEに生成してください。
            考察: {self.discussion}
            """
        )
        self.conclusion = query_model(prompt_conc)

    def report_writing(self):
        print("[ステップ5] レポート作成: 論文形式のレポートをまとめ中…")
        self.data_preparation()
        self.generate_discussion_and_conclusion()
        report = textwrap.dedent(f"""
        # 実験報告書: {self.topic}

        ## 1. はじめに
        文献レビューと実験計画を踏まえ、本研究では『{self.topic}』の調査を行った。

        ## 2. 文献レビュー
        {self.lit_summary}

        ## 3. 実験計画
        {self.plan_text}

        ## 4. 追加文献レビュー: 結果・考察
        {self.additional_summary}

        ## 5. 実験結果 (Results)
        {self.exp_results}

        ## 6. 考察 (Discussion)
        {self.discussion}

        ## 8. 結論 (Conclusion)
        {self.conclusion}
        """
        )
        (self.lab_dir / "report.md").write_text(report)
        print(f"レポートを保存 → {self.lab_dir / 'report.md'}")

    def perform(self):
        start = time.time()
        self.literature_review()
        self.plan_formulation()
        self.additional_literature_review()
        self.report_writing()
        print(
            f"\n全フェーズ完了: 実行ディレクトリ → {self.lab_dir} (所要時間: {time.time()-start:.1f} 秒)"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=False, help="研究トピック")
    ap.add_argument("--n", type=int, default=5, help="論文数(arXiv/S2)")
    args = ap.parse_args()
    topic = args.topic or input("研究トピックを入力してください: ")
    wf = LaboratoryWorkflow(topic=topic, n_papers=args.n)
    wf.perform()

if __name__ == "__main__":
    main()
