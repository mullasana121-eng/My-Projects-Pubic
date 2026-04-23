import tkinter as tk
from tkinter import messagebox

def convert_temperature():
    try:
        temp = float(entry_temp.get())
        # Check which conversion is selected via the radio buttons
        if var.get() == 1:  # C to F
            result = (temp * 9/5) + 32
            lbl_result.config(text=f"{temp}°C is {result:.2f}°F")
        elif var.get() == 2:  # F to C
            result = (temp - 32) * 5/9
            lbl_result.config(text=f"{temp}°F is {result:.2f}°C")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number!")

# --- UI Setup ---
root = tk.Tk()
root.title("Time Traveler's Temp Converter")
root.geometry("350x250")

# Title Label
tk.Label(root, text="Universal Temperature Converter", font=("Arial", 12, "bold")).pack(pady=10)

# Entry Field
tk.Label(root, text="Enter Temperature:").pack()
entry_temp = tk.Entry(root)
entry_temp.pack(pady=5)

# Radio Buttons for Selection
var = tk.IntVar(value=1)
tk.Radiobutton(root, text="Celsius to Fahrenheit", variable=var, value=1).pack()
tk.Radiobutton(root, text="Fahrenheit to Celsius", variable=var, value=2).pack()

# Convert Button
btn_convert = tk.Button(root, text="Convert Now", command=convert_temperature, bg="#4CAF50", fg="white")
btn_convert.pack(pady=15)

# Result Display
lbl_result = tk.Label(root, text="", font=("Arial", 11, "italic"))
lbl_result.pack()

root.mainloop()