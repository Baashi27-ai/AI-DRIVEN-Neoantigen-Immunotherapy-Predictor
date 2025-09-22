path = r"C:\Neo_Antigen_Moonshot\Phase 2 Neoantigen Prediction Pipeline\scripts\validate_phase2_inputs_clean.py"

lines = [
    'print(">>> Phase 2 Input Validation — Minimal test")\n',
    'print("[OK] Script executed correctly.")\n',
    'def main():\n',
    '    print("Main function block works!")\n',
    'if _name_ == "_main_":\n',
    '    main()\n'
]

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"[OK] Overwrote clean validator at: {path}")