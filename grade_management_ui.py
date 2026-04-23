import tkinter as tk
from tkinter import messagebox, simpledialog

students = {}

def add_student():
    name = simpledialog.askstring("Add Student", "Enter student name:")
    if name:
        if name not in students:
            students[name] = []
            messagebox.showinfo("Success", "Student added!")
        else:
            messagebox.showwarning("Warning", "Student already exists.")

def add_grade():
    name = simpledialog.askstring("Add Grade", "Enter student name:")
    if name in students:
        try:
            grade = float(simpledialog.askstring("Add Grade", "Enter grade:"))
            students[name].append(grade)
            messagebox.showinfo("Success", "Grade added!")
        except:
            messagebox.showerror("Error", "Invalid grade input.")
    else:
        messagebox.showerror("Error", "Student not found.")

def view_student():
    name = simpledialog.askstring("View Student", "Enter student name:")
    if name in students:
        grades = students[name]
        if grades:
            avg = sum(grades) / len(grades)
            messagebox.showinfo("Student Info", f"Grades: {grades}\nAverage: {round(avg,2)}")
        else:
            messagebox.showinfo("Student Info", "No grades available.")
    else:
        messagebox.showerror("Error", "Student not found.")

def class_average():
    total = 0
    count = 0
    for grades in students.values():
        total += sum(grades)
        count += len(grades)

    if count > 0:
        avg = total / count
        messagebox.showinfo("Class Average", f"Average: {round(avg,2)}")
    else:
        messagebox.showinfo("Class Average", "No grades available.")

root = tk.Tk()
root.title("Grade Management System")
root.geometry("400x300")

title = tk.Label(root, text="Grade Management System", font=("Arial", 14, "bold"))
title.pack(pady=10)

btn1 = tk.Button(root, text="Add Student", width=20, command=add_student)
btn1.pack(pady=5)

btn2 = tk.Button(root, text="Add Grade", width=20, command=add_grade)
btn2.pack(pady=5)

btn3 = tk.Button(root, text="View Student", width=20, command=view_student)
btn3.pack(pady=5)

btn4 = tk.Button(root, text="Class Average", width=20, command=class_average)
btn4.pack(pady=5)

btn5 = tk.Button(root, text="Exit", width=20, command=root.quit)
btn5.pack(pady=10)

root.mainloop()
