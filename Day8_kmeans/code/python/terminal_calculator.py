"""
Terminal Calculator - A6 Practical Task
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
    print("  5. Exit")
    print("=" * 40)

def get_choice():
    """Get user's operation choice."""
    while True:
        try:
            choice = int(input("Enter your choice (1-5): "))
            if 1 <= choice <= 5:
                return choice
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def perform_calculation(choice, a, b):
    """Perform the selected calculation."""
    if choice == 1:
        return add(a, b)
    elif choice == 2:
        return subtract(a, b)
    elif choice == 3:
        return multiply(a, b)
    elif choice == 4:
        return divide(a, b)
    else:
        return None

def main():
    """Main calculator loop."""
    print("Welcome to the Terminal Calculator!")
    
    while True:
        display_menu()
        choice = get_choice()
        
        if choice == 5:
            print("Thank you for using the calculator. Goodbye!")
            break
        
        a, b = get_numbers()
        result = perform_calculation(choice, a, b)
        
        if result is not None:
            print(f"\nResult: {result}")
        else:
            print("Invalid operation.")

if __name__ == "__main__":
    main()