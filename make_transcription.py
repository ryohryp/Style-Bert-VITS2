import os
import glob
import whisper

# 1. 分割されたWAVファイルが入っているフォルダを指定（環境に合わせて書き換えてください）
# ※先ほどの出力フォルダを指定します
TARGET_DIR = r"I:\04_develop\Style-Bert-VITS2-1\Data\umekin\raw" 

# 2. 成果物となるテキストの保存先
OUTPUT_FILE = os.path.join(TARGET_DIR, "transcription.txt")
# Style-Bert-VITS2用の学習リストファイル
ESD_FILE = os.path.join(os.path.dirname(TARGET_DIR), "esd.list")

def main():
    print("Whisperモデル（base）を読み込んでいます（CPUモード）...")
    # baseモデルは軽量で日本語精度も高く、CPUでも爆速で動きます
    model = whisper.load_model("base", device="cpu")
    
    # フォルダ内のWAVファイルをソートして取得
    wav_files = sorted(glob.glob(os.path.join(TARGET_DIR, "*.wav")))
    
    if not wav_files:
        print(f"エラー: 指定されたフォルダにWAVファイルが見つかりません: {TARGET_DIR}")
        return

    print(f"{len(wav_files)}個の音声ファイルの文字起こしを開始します...")
    
    # 親ディレクトリ名（例: umekin）を話者名/モデル名として使用
    model_name = os.path.basename(os.path.dirname(TARGET_DIR))
    
    lines = []
    esd_lines = []
    for i, wav_path in enumerate(wav_files):
        filename = os.path.basename(wav_path)
        
        # Whisperで耳コピを実行するために、librosaで16000Hzでロード（ffmpeg不要にするため）
        import librosa
        audio, _ = librosa.load(wav_path, sr=16000)
        result = model.transcribe(audio, language="ja")
        text = result["text"].strip()
        
        # 簡易フォーマット「ファイル名|テキスト」
        line = f"{filename}|{text}"
        lines.append(line)
        
        # Style-Bert-VITS2フォーマット「ファイル名|モデル名|言語|テキスト」
        esd_line = f"{filename}|{model_name}|JP|{text}"
        esd_lines.append(esd_line)
        
        print(f"[{i+1}/{len(wav_files)}] {line}")
        
    # transcription.txt に書き出し（UTF-8）
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    # esd.list に書き出し（UTF-8、最後に改行を追加）
    with open(ESD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(esd_lines) + "\n")
        
    print(f"\n成功しました！文字起こしが完了しました:")
    print(f" - 簡易形式: {OUTPUT_FILE}")
    print(f" - Style-Bert-VITS2形式: {ESD_FILE}")

if __name__ == "__main__":
    main()