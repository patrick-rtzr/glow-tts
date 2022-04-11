from text import mixed_g2p, text_to_sequence

with open("data/valid_filelist.txt", encoding="utf-8") as f, open("data/valid.txt", "w", encoding="utf-8") as f2:
    for line in f.readlines():
        w, l = line.strip().split("|")
        l = text_to_sequence(l)
        o = w + "|" + l
        f2.write(f"{o}\n")

