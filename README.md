# SurveyFlow

[English](#english) | [中文](#中文) | [日本語](#日本語) | [한국어](#한국어)

## English

SurveyFlow is a local graphical tool for literature discovery, abstract
enrichment, AI-assisted screening, human review, and citation snowballing.

### Fastest Start

Download `SurveyFlow-quickstart.zip` from the project's GitHub Releases page and
extract it, then:

**Windows:** double-click `start.bat`.

**macOS / Linux:** open a terminal in the extracted folder and run:

```bash
sh start.sh
```

Open <http://localhost:8501>. On the first start, the launcher installs a private
copy of `uv`, Python, and the required packages. Nothing is installed globally.
Internet access is required during the first start and while searching papers.

To stop SurveyFlow, return to the launcher terminal and press `Ctrl+C`. On
Windows, closing the launcher window also stops the app.

### Docker Start

When Docker Desktop is installed, run:

```bash
docker compose up --build
```

Open <http://localhost:8501>. To stop it, press `Ctrl+C`, then run:

```bash
docker compose down
```

### Use the App

1. Select the interface language in the sidebar.
2. Create a project, choose the research field and recommended literature sources,
   then enter the research question, years, and keyword groups.
3. On **Scope**, adjust the source multi-select and check the Boolean query preview.
4. Optionally add an OpenAI API key on **AI settings** and choose separate models
   for screening, paper Q&A, and corpus classification.
5. Start retrieval on **Run center**. The background run, paper count, and progress
   remain visible after you leave the page and return.
6. On **Add papers**, search a known title or enter its metadata manually. Synchronize
   additions before creating the review queue.
7. Prepare the round with AI screening or human-only screening.
8. Make the final decision for every paper on **Manual review**.
9. Run citation expansion on **Snowball**, then export from **Results**.
10. On **AI research**, upload a PDF to ask questions with saved conversation
    memory, or classify the whole final corpus using your own criteria. Leave the
    criteria empty to let the model propose a taxonomy.

Live search connectors are available for DBLP, OpenAlex, Crossref, arXiv, and
PubMed. OpenAIRE and Europeana appear as planned specialist sources and cannot
yet be selected. Humanities, arts, design, and architecture projects default to
broad scholarly indexes and can use manual additions for books, catalogues, or
local publications missed by those indexes.

AI recommendations never replace human decisions. Projects and results are
stored in `data/app_projects/`; saved API keys are stored separately in
`.secrets/app_projects/`. These folders persist when Docker is rebuilt.
Uploaded PDFs, conversations, taxonomies, classifications, and analysis reports
are also stored locally inside the project folder.
When an AI action is run, the required paper text or PDF is sent to the configured
API endpoint; users should only upload documents they are permitted to process.

Already have `uv`? Run `uv sync --no-dev` and `uv run vnn-survey-app`.

## 中文

SurveyFlow 是一个本地运行的图形化文献综述工具，支持论文检索、摘要补全、
AI 辅助筛选、人工审阅和引用滚雪球。

### 最快启动方式

从项目的 GitHub Releases 页面下载并解压 `SurveyFlow-quickstart.zip`，然后：

**Windows：** 双击 `start.bat`。

**macOS / Linux：** 在解压目录打开终端并运行：

```bash
sh start.sh
```

打开 <http://localhost:8501>。首次启动时，脚本会在项目目录内自动安装独立的
`uv`、Python 和所需依赖，不会修改系统 Python。首次启动和检索论文时需要联网。

停止 SurveyFlow 时，回到启动程序所在的终端并按 `Ctrl+C`。在 Windows 上也可以
直接关闭启动程序的窗口。

### 使用 Docker 启动

如果已经安装 Docker Desktop，运行：

```bash
docker compose up --build
```

打开 <http://localhost:8501>。停止时先按 `Ctrl+C`，然后运行：

```bash
docker compose down
```

### 平台使用流程

1. 在侧边栏选择界面语言。
2. 创建项目，选择研究领域和推荐数据源，然后填写研究问题、年份和关键词组。
3. 在**研究范围**页面调整数据源多选并检查布尔查询逻辑。
4. 如需 AI 功能，在 **AI 设置**中填写 OpenAI API 密钥，并分别选择筛选、论文问答和
   文献集分类模型。
5. 在**运行中心**开始检索。任务会在后台继续，离开页面后再次返回仍可查看当前阶段、
   进度和已收集论文数量。
6. 在**添加论文**中按标题查找已知论文，或手工填写元数据；建立审阅队列前同步补录结果。
7. 选择 AI 辅助筛选或仅人工筛选，生成审阅队列。
8. 在**人工审阅**中对每篇论文作出最终决定。
9. 在**滚雪球检索**中扩展引用，最后在**结果**页面导出。
10. 在 **AI 研究**中上传 PDF 并进行带本地对话记忆的问答，或按自行提供的标准对
    整个最终文献集分类；不填写标准时由模型提出分类体系。

当前可实时检索 DBLP、OpenAlex、Crossref、arXiv 和 PubMed。OpenAIRE 与
Europeana 会显示为计划接入的专业数据源，目前不能选择。人文、艺术、设计与建筑项目
默认使用综合性学术索引，并可通过人工补录加入其中遗漏的书籍、图录或地方出版物。

AI 只提供建议，不能代替人工决定。项目与结果保存在 `data/app_projects/`，
保存的 API 密钥单独位于 `.secrets/app_projects/`。重新构建 Docker 不会删除这些数据。
上传的 PDF、对话、分类体系、分类结果和分析报告同样保存在对应项目目录中。
运行 AI 功能时，所需的论文文本或 PDF 会发送到配置的 API 端点；请仅上传获准处理的文档。

如果已经安装 `uv`，直接运行 `uv sync --no-dev` 和
`uv run vnn-survey-app` 即可。

## 日本語

SurveyFlow は、文献検索、抄録の補完、AI 支援スクリーニング、手動レビュー、
引用スノーボール検索に対応したローカル実行の文献レビュー用 GUI ツールです。

### 最も簡単な起動方法

プロジェクトの GitHub Releases ページから `SurveyFlow-quickstart.zip` を
ダウンロードして展開し、次のように起動します。

**Windows：** `start.bat` をダブルクリックします。

**macOS / Linux：** 展開したフォルダーでターミナルを開き、次を実行します。

```bash
sh start.sh
```

<http://localhost:8501> を開きます。初回起動時に、ランチャーがプロジェクト内へ
専用の `uv`、Python、必要なパッケージを自動でインストールします。システムの
Python は変更しません。初回起動時と論文検索時にはインターネット接続が必要です。

SurveyFlow を停止するには、起動に使用したターミナルへ戻って `Ctrl+C` を押します。
Windows ではランチャーのウィンドウを閉じても停止できます。

### Docker で起動

Docker Desktop がインストールされている場合は、次を実行します。

```bash
docker compose up --build
```

<http://localhost:8501> を開きます。停止するには `Ctrl+C` を押してから、次を実行します。

```bash
docker compose down
```

### アプリの使用手順

1. サイドバーで表示言語を選択します。
2. プロジェクトを作成し、研究分野と推奨データソースを選んでから、研究課題、対象年、
   キーワード・グループを入力します。
3. **研究範囲**ページでデータソースの複数選択を調整し、ブール検索式を確認します。
4. AI 機能を利用する場合は、**AI 設定**で OpenAI API キーを入力し、スクリーニング、
   論文 Q&A、コーパス分類に使用するモデルを個別に選択します。
5. **実行センター**で検索を開始します。他のページへ移動して戻っても、バックグラウンド
   実行、現在の工程、進捗、収集済み論文数を確認できます。
6. **論文を追加**で既知のタイトルを検索するか、メタデータを手動入力します。
   レビュー待ち行列を作成する前に追加内容を同期してください。
7. AI 支援または手動のみのスクリーニングを選択し、レビュー待ち行列を作成します。
8. **手動レビュー**で各論文の最終判定を入力します。
9. **スノーボール検索**で引用を展開し、最後に**結果**ページから出力します。
10. **AI 研究**で PDF をアップロードしてローカル保存される会話履歴付き Q&A を行うか、
    独自の基準で最終コーパス全体を分類します。基準を空欄にするとモデルが分類体系を提案します。

DBLP、OpenAlex、Crossref、arXiv、PubMed はライブ検索に対応しています。
OpenAIRE と Europeana は専門データソースとして対応予定と表示され、まだ選択できません。
人文学、芸術、デザイン、建築のプロジェクトでは幅広い学術索引を既定値とし、索引から
漏れた書籍、カタログ、地域出版物を手動で追加できます。

AI の提案が人による最終判定を置き換えることはありません。プロジェクトと結果は
`data/app_projects/` に保存され、API キーは `.secrets/app_projects/` に分けて
保存されます。Docker イメージを再構築しても、これらのデータは維持されます。
アップロードした PDF、会話、分類体系、分類結果、分析レポートもプロジェクト内に保存されます。
AI 機能の実行時には必要な論文テキストまたは PDF が設定済み API へ送信されます。
処理の許可を得た文書のみをアップロードしてください。

既に `uv` がインストールされている場合は、`uv sync --no-dev` と
`uv run vnn-survey-app` を実行できます。

## 한국어

SurveyFlow는 문헌 검색, 초록 보강, AI 보조 선별, 수동 검토 및 인용
스노우볼링을 지원하는 로컬 그래픽 문헌 검토 도구입니다.

### 가장 빠른 시작 방법

프로젝트의 GitHub Releases 페이지에서 `SurveyFlow-quickstart.zip`을 다운로드하고
압축을 푼 후 다음과 같이 실행합니다.

**Windows:** `start.bat`을 두 번 클릭합니다.

**macOS / Linux:** 압축을 푼 폴더에서 터미널을 열고 실행합니다.

```bash
sh start.sh
```

<http://localhost:8501>을 엽니다. 처음 실행할 때 런처가 프로젝트 폴더 안에
전용 `uv`, Python 및 필수 패키지를 자동으로 설치합니다. 시스템 Python은
변경하지 않습니다. 최초 실행과 논문 검색에는 인터넷 연결이 필요합니다.

SurveyFlow를 종료하려면 실행에 사용한 터미널로 돌아가 `Ctrl+C`를 누릅니다.
Windows에서는 실행 창을 닫아도 앱이 종료됩니다.

### Docker로 시작

Docker Desktop이 설치되어 있다면 다음을 실행합니다.

```bash
docker compose up --build
```

<http://localhost:8501>을 엽니다. 종료하려면 `Ctrl+C`를 누른 후 다음을 실행합니다.

```bash
docker compose down
```

### 앱 사용 순서

1. 사이드바에서 인터페이스 언어를 선택합니다.
2. 프로젝트를 만들고 연구 분야와 추천 데이터 소스를 선택한 다음 연구 질문, 연도 및
   키워드 그룹을 입력합니다.
3. **연구 범위** 페이지에서 데이터 소스 다중 선택을 조정하고 불리언 검색식을 확인합니다.
4. AI 기능이 필요하면 **AI 설정**에서 OpenAI API 키를 입력하고 선별, 논문 Q&A,
   문헌 집합 분류에 사용할 모델을 각각 선택합니다.
5. **실행 센터**에서 검색을 시작합니다. 다른 페이지에 다녀와도 백그라운드 실행,
   현재 단계, 진행률과 수집된 논문 수를 계속 확인할 수 있습니다.
6. **논문 추가**에서 알고 있는 제목을 검색하거나 메타데이터를 직접 입력합니다. 검토
   대기열을 만들기 전에 추가 항목을 동기화하세요.
7. AI 보조 또는 수동 전용 선별을 선택하여 검토 대기열을 만듭니다.
8. **수동 검토**에서 모든 논문의 최종 결정을 입력합니다.
9. **스노우볼 검색**으로 인용을 확장한 후 **결과**에서 내보냅니다.
10. **AI 연구**에서 PDF를 업로드하여 로컬 대화 기록이 유지되는 Q&A를 사용하거나,
    직접 제공한 기준으로 전체 최종 문헌 집합을 분류합니다. 기준을 비워 두면 모델이
    분류 체계를 제안합니다.

DBLP, OpenAlex, Crossref, arXiv 및 PubMed는 실시간 검색을 지원합니다. OpenAIRE와
Europeana는 전문 데이터 소스로 지원 예정 상태이며 아직 선택할 수 없습니다. 인문학,
예술, 디자인, 건축 프로젝트는 광범위한 학술 색인을 기본으로 사용하고 누락된 도서,
전시 도록 또는 지역 출판물을 수동으로 추가할 수 있습니다.

AI 제안은 사람의 최종 결정을 대체하지 않습니다. 프로젝트와 결과는
`data/app_projects/`에 저장되고 API 키는 `.secrets/app_projects/`에 별도로
저장됩니다. Docker 이미지를 다시 빌드해도 이 데이터는 유지됩니다.
업로드한 PDF, 대화, 분류 체계, 분류 결과 및 분석 보고서도 프로젝트 폴더에 로컬로 저장됩니다.
AI 기능을 실행하면 필요한 논문 텍스트 또는 PDF가 설정된 API 엔드포인트로 전송됩니다.
처리가 허용된 문서만 업로드하세요.

이미 `uv`가 설치되어 있다면 `uv sync --no-dev`와
`uv run vnn-survey-app`을 실행하면 됩니다.

---

Maintainers can rebuild the downloadable package with `make package`.
