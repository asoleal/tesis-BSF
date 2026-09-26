from pathlib import Path
p = Path("patch_paso1.py")
s = p.read_text()
s = s.replace("assert t.count(old_t) == 1, t.count(old_t)\nm.write_text(t.replace(old_t, new_t))",
              "if t.count(old_t) == 1:\n    t = t.replace(old_t, new_t)\nm.write_text(t)")
s = s.replace("s = s[:i0] + nuevo + s[i:]", "s = s[:i0] + nuevo + s[i1:]")
p.write_text(s)
print("fix ok")
