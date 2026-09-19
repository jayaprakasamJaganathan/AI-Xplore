"""
Terminal Calculator - A7 Practical Task
=======================================
Calculator with add, subtract, multiply, divide functions.
Execute via: python terminal_calculator.py
"""

def add(a, b):
    """Add two numbers."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


def multiply(a, b):
    """Multiply two numbers."""
    return a * b


def divide(a, b):
    """Divide a by b. Handles division by zero."""
    if b == 0:
        return "Error: Division by zero"
    return a / b


def get_numbers():
    """Get two numbers from user input."""
    while True:
        try:
            user_input = input("Enter two numbers separated by space: ")
            a, b = map(float, user_input.split())
            return a, b
        except ValueError:
            print("Invalid input. Please enter two numbers separated by space.")


def display_menu():
    """Display operation menu."""
    print("\n" + "=" * 40)
    print("CALCULATOR")
    print("=" * 40)
    print("Operations:")
    print("  1. Add (+)")
    print("  2. Subtract (-)")
    print("  3. Multiply (*)")
    print("  4. Divide (/)")
    print("  5. All operations")
    print("  6. Exit")
    print("-" * 40)


def get_choice():
    """Get user's operation choice."""
    while True:
        try:
            choice = int(input("Enter choice (1-6): "))
            if 1 <= choice <= 6:
                return choice
            print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def perform_calculation(a, b, choice):
    """Perform the selected calculation."""
    if choice == 1:
        result = add(a, b)
        print(f"\nResult: {a} + {b} = {result}")
    elif choice == 2:
        result = subtract(a, b)
        print(f"\nResult: {a} - {b} = {result}")
    elif choice == 3:
        result = multiply(a, b)
        print(f"\nResult: {a} * {b} = {result}")
    elif choice == 4:
        result = divide(a, b)
        print(f"\nResult: {a} / {b} = {result}")
    elif choice == 5:
        print(f"\nAll Operations for {a} and {b}:")
        print(f"  {a} + {b} = {add(a, b)}")
        print(f"  {a} - {b} = {subtract(a, b)}")
        print(f"  {a} * {b} = {multiply(a, b)}")
        print(f"  {a} / {b} = {divide(a, b)}")


def main():
    """Main calculator loop."""
    print("Welcome to the Terminal Calculator!")
    
    while True:
        display_menu()
        choice = get_choice()
        
        if choice == 6:
            print("\nThank you for using the calculator. Goodbye!")
            break
        
        a, b = get_numbers()
        perform_calculation(a, b, choice)


if __name__ == "__main__":
    main()