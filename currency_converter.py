import tkinter as tk
from tkinter import ttk, messagebox
import requests

# Function to get exchange rate
def convert_currency():
    try:
        amount = float(amount_entry.get())
        from_curr = from_currency.get()
        to_curr = to_currency.get()

        url = f"https://api.exchangerate-api.com/v4/latest/{from_curr}"
        response = requests.get(url)
        data = response.json()

        rate = data["rates"][to_curr]
        result = amount * rate

        result_label.config(text=f"{amount} {from_curr} = {round(result, 2)} {to_curr}")

    except Exception as e:
        messagebox.showerror("Error", "Something went wrong!\nCheck input or internet.")

# Reverse currencies
def reverse_currency():
    from_curr = from_currency.get()
    to_curr = to_currency.get()

    from_currency.set(to_curr)
    to_currency.set(from_curr)

# UI Window
root = tk.Tk()
root.title("Currency Converter")
root.geometry("400x300")

# Title
title = tk.Label(root, text="Real-Time Currency Converter", font=("Arial", 14, "bold"))
title.pack(pady=10)

# Amount input
amount_label = tk.Label(root, text="Enter Amount:")
amount_label.pack()
amount_entry = tk.Entry(root)
amount_entry.pack(pady=5)

# Currency list
currencies = ["USD", "INR", "EUR", "JPY", "GBP", "AUD", "CAD"]

# From currency
from_currency = tk.StringVar(value="USD")
from_label = tk.Label(root, text="From Currency:")
from_label.pack()
from_menu = ttk.Combobox(root, textvariable=from_currency, values=currencies)
from_menu.pack(pady=5)

# To currency
to_currency = tk.StringVar(value="INR")
to_label = tk.Label(root, text="To Currency:")
to_label.pack()
to_menu = ttk.Combobox(root, textvariable=to_currency, values=currencies)
to_menu.pack(pady=5)

# Buttons
convert_btn = tk.Button(root, text="Convert", command=convert_currency)
convert_btn.pack(pady=5)

reverse_btn = tk.Button(root, text="Reverse 🔄", command=reverse_currency)
reverse_btn.pack(pady=5)

# Result
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Run app
root.mainloop()