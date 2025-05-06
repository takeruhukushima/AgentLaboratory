# ai_lab_repo.py — AgentLaboratory
# =============================================================
#  * Phase 1: Literature Review
#  * Phase 2: Plan Formulation
#  * Phase 3: Data Preparation (Iris subset)
#  * Phase 4: Running Experiments (Logistic Regression)
#  * Phase 5: Report Writing (生成された結果を日本語の論文風にまとめる)
# --------------------------------------------------------------------

from __future__ import annotations

import os, argparse, time, textwrap
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.genai as genai           # using google-genai SDK
import arxiv

# Load API key
load_dotenv()
_API_KEY = os.getenv("GOOGLE_API_KEY")
if not _API_KEY:
    raise EnvironmentError("環境変数 GOOGLE_API_KEY が設定されていません。")
_CLIENT = genai.Client(api_key=_API_KEY)

DEFAULT_MODEL = "gemini-2.0-flash"
RESEARCH_DIR_PATH = "MATH_research_dir"

# ────────────────────────────────────────────────────────────────
# helper wrappers
# ----------------------------------------------------------------

def query_model(prompt: str, model_str: str = DEFAULT_MODEL) -> str:
    """Gemini を呼び出してテキスト応答を取得（日本語プロンプト可）。"""
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

# ════════════════════════════════════════════════════════════════
# LaboratoryWorkflow
# ════════════════════════════════════════════════════════════════

class LaboratoryWorkflow:
    def __init__(self, *, topic: str, n_papers: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lab_dir = Path(RESEARCH_DIR_PATH) / f"run_{ts}"
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        self.topic = topic
        self.n_papers = n_papers
        self.searcher = ArxivSearch()
        self.lit_summary = ""
        self.plan_text = ""
        self.dataset_path: Path | None = None
        self.exp_results = ""

    # -----------------------------------------------------------
    def literature_review(self):
        print(f"[ステップ1] 文献レビュー: '{self.topic}' を arXiv から検索中…")
        papers = self.searcher.search(self.topic, self.n_papers)
        bullets = []
        for i, p in enumerate(papers, 1):
            prompt = textwrap.dedent(
                f"""
                以下の論文の貢献、手法、結果を日本語で3文以内に要約してください。
                論文タイトル: {p.title}
                """
            )
            summ = query_model(prompt)
            bullets.append(f"- **{p.title}** — {summ}")
            print(f"[{i}] {p.title}\n→ {summ}\n")
        md = "### 文献レビュー\n" + "\n".join(bullets)
        (self.lab_dir / "literature_review.md").write_text(md)
        self.lit_summary = md
        print(f"文献レビュー結果を保存 → {self.lab_dir / 'literature_review.md'}")

    # -----------------------------------------------------------
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
        plan = query_model(system_prompt)
        (self.lab_dir / "plan.md").write_text(plan)
        self.plan_text = plan
        print(f"実験計画を保存 → {self.lab_dir / 'plan.md'}")

    # -----------------------------------------------------------
    def data_preparation(self):
        print("[ステップ3] データ準備: Iris サブセットを生成中…")
        from sklearn.datasets import load_iris
        import pandas as pd
        iris = load_iris(as_frame=True)
        df = iris.frame.sample(100, random_state=0)
        self.dataset_path = self.lab_dir / "iris_subset.csv"
        df.to_csv(self.dataset_path, index=False)
        print(f"データセットを保存 → {self.dataset_path}")

    # -----------------------------------------------------------
    def running_experiments(self):
        print("[ステップ4] 実験実行: ロジスティック回帰モデルを訓練・評価中…")
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        if not self.dataset_path:
            raise RuntimeError("データセットが用意されていません。")
        df = pd.read_csv(self.dataset_path)
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        self.exp_results = f"保持データセットに対する精度: {acc:.3f}"
        (self.lab_dir / "results.txt").write_text(self.exp_results)
        print(self.exp_results)
        print(f"実験結果を保存 → {self.lab_dir / 'results.txt'}")

    # -----------------------------------------------------------
    def report_writing(self):
        print("[ステップ5] レポート作成: 結果を論文風にまとめ中…")
        report = textwrap.dedent(
            f"""
            # 実験報告書: {self.topic}

            ## はじめに
            文献レビューと実験計画をもとに『{self.topic}』について以下の実験を実施しました。

            ## 文献レビュー概要
            {self.lit_summary}

            ## 実験計画
            {self.plan_text}

            ## データ準備
            Iris データセットのサブセット (100 サンプル) を使用しました。

            ## 実験結果
            {self.exp_results}

            ## 結論
            実験により、Iris データセット上でモデルは一定の精度を示しました。今後は実際のスピノーダル分解データを用いた検証が必要です。
            """
        )
        (self.lab_dir / "report.md").write_text(report)
        print(f"レポートを保存 → {self.lab_dir / 'report.md'}")

    # -----------------------------------------------------------
    def perform(self):
        start = time.time()
        self.literature_review()
        self.plan_formulation()
        self.data_preparation()
        self.running_experiments()
        self.report_writing()
        print(f"\n全フェーズ完了: 実行ディレクトリ → {self.lab_dir} (所要時間: {time.time()-start:.1f} 秒)")

# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=False, help="研究トピック")
    ap.add_argument("--n", type=int, default=5, help="arXiv 論文数")
    args = ap.parse_args()

    topic = args.topic or input("研究トピックを入力してください: ")
    wf = LaboratoryWorkflow(topic=topic, n_papers=args.n)
    wf.perform()

if __name__ == "__main__":
    main()
