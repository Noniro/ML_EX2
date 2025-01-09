import math


def check_inequality(s):
    # חישוב צד שמאל של האי-שוויון
    left = (s ** 8 * (2 ** (8 * s))) / (16 * math.e * (s ** 2))

    # חישוב צד ימין של האי-שוויון
    right = math.log2(s * (2 ** s))

    # בדיקה האם האי-שוויון מתקיים
    return left < right


# בדיקת ערכי s
for s in range(1, 101):  # נניח טווח מ-1 עד 100
    if not check_inequality(s):
        print(f"Inequality does not hold for s = {s}")
