code = """print(">>> Phase 2 Input Validation — Minimal test")
print("[OK] Script executed correctly.")

if _name_ == "_main_":
    main()
"""

path = r"C:\Neo_Antigen_Moonshot\Phase 2 Neoantigen Prediction Pipeline\scripts\validate_phase2_inputs_clean.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print(f"[OK] Wrote clean script -> {path}")