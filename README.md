# Tetris AI
このプロジェクトは Gemini CLI を使ったAIコード生成のテストと、Git/GitHub 学習を兼ねて作成しました。  
ほとんどのコードは生成AIが書いたものであり、自分はそれを動かしながら理解・修正し、公開用に整えています。  

## 特徴
- **対戦モード**
  - Human vs AI
  - AI vs AI
- **ゲーム機能**
  - ライン消去に応じた攻撃
  - Rキーでリスタート / ESCキーで終了

## 技術スタック
- 言語: Python 3.12.2
- ライブラリ: pygame
- 開発環境: Gemini CLI + Git/GitHub

## 実行方法
1. リポジトリをクローン
   ```bash
   git clone https://github.com/koki01150124/tetris-ai.git
2. tetris-aiに移動
   ```bash
   cd tetris-ai
4. 依存ライブラリをインストール
   ```bash
   pip install -r requirements.txt
5. 実行
   ```bash
   python tetris_ai.py
