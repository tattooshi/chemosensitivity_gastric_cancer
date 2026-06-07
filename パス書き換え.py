import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
import torch
import shutil
import pprint


class PthEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch .pth editor")

        self.pth_path = None
        self.data = None

        frame = tk.Frame(root)
        frame.pack(fill="x", padx=10, pady=10)

        tk.Button(frame, text=".pthを開く", command=self.open_pth).pack(side="left")
        tk.Button(frame, text="再表示", command=self.refresh_view).pack(side="left", padx=5)
        tk.Button(frame, text="上書き保存", command=self.save_pth).pack(side="left", padx=5)

        replace_frame = tk.LabelFrame(root, text="文字列一括置換")
        replace_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(replace_frame, text="置換前").grid(row=0, column=0, sticky="w")
        tk.Label(replace_frame, text="置換後").grid(row=1, column=0, sticky="w")

        self.old_entry = tk.Entry(replace_frame, width=100)
        self.old_entry.grid(row=0, column=1, padx=5, pady=3)

        self.new_entry = tk.Entry(replace_frame, width=100)
        self.new_entry.grid(row=1, column=1, padx=5, pady=3)

        tk.Button(replace_frame, text="置換実行", command=self.replace_text).grid(
            row=0, column=2, rowspan=2, padx=5
        )

        self.text = scrolledtext.ScrolledText(root, width=140, height=35)
        self.text.pack(fill="both", expand=True, padx=10, pady=10)

    def open_pth(self):
        path = filedialog.askopenfilename(
            title=".pthファイルを選択",
            filetypes=[("PyTorch files", "*.pth *.pt"), ("All files", "*.*")]
        )

        if not path:
            return

        self.pth_path = Path(path)

        try:
            self.data = torch.load(self.pth_path, map_location="cpu", weights_only=False)
        except Exception as e:
            messagebox.showerror("エラー", f"読み込みに失敗しました:\n{e}")
            return

        self.refresh_view()

    def refresh_view(self):
        if self.data is None:
            return

        self.text.delete("1.0", tk.END)

        info = []
        info.append(f"file: {self.pth_path}\n")
        info.append(f"type: {type(self.data)}\n\n")

        if isinstance(self.data, dict):
            info.append("keys:\n")
            for k in self.data.keys():
                info.append(f"  - {k}: {type(self.data[k])}\n")

            info.append("\n\nfull content:\n")
            info.append(pprint.pformat(self.data, width=160))
        else:
            info.append(pprint.pformat(self.data, width=160))

        self.text.insert(tk.END, "".join(info))

    def replace_text(self):
        if self.data is None:
            messagebox.showwarning("注意", "先に.pthファイルを開いてください")
            return

        old = self.old_entry.get()
        new = self.new_entry.get()

        if old == "":
            messagebox.showwarning("注意", "置換前の文字列が空です")
            return

        self.data = self.recursive_replace(self.data, old, new)
        self.refresh_view()
        messagebox.showinfo("完了", "置換しました。保存するには上書き保存を押してください。")

    def recursive_replace(self, x, old, new):
        if isinstance(x, str):
            return x.replace(old, new)
        elif isinstance(x, list):
            return [self.recursive_replace(v, old, new) for v in x]
        elif isinstance(x, tuple):
            return tuple(self.recursive_replace(v, old, new) for v in x)
        elif isinstance(x, dict):
            return {
                self.recursive_replace(k, old, new): self.recursive_replace(v, old, new)
                for k, v in x.items()
            }
        else:
            return x

    def save_pth(self):
        if self.data is None or self.pth_path is None:
            messagebox.showwarning("注意", "保存するデータがありません")
            return

        backup_path = self.pth_path.with_suffix(".backup.pth")

        try:
            shutil.copy2(self.pth_path, backup_path)
            torch.save(self.data, self.pth_path)
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")
            return

        messagebox.showinfo(
            "保存完了",
            f"上書き保存しました。\n\nバックアップ:\n{backup_path}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = PthEditorApp(root)
    root.mainloop()