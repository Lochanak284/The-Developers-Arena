

from colorama import Fore, Style, init

init(autoreset=True)

print(Fore.CYAN + "📘 Welcome to the Student Grade Calculator!\n")


try:
    marks = float(input(Fore.YELLOW + "Enter your marks (0 - 100): " + Fore.WHITE))

    
    if marks < 0 or marks > 100:
        print(Fore.RED + "❌ Please enter a valid mark between 0 and 100.")
    else:
        
        if marks >= 90:
            grade = "A+"
            message = "🌟 Outstanding performance! Keep shining!"
            color = Fore.GREEN
        elif marks >= 80:
            grade = "A"
            message = "💪 Great job! You’re doing amazing."
            color = Fore.LIGHTGREEN_EX
        elif marks >= 70:
            grade = "B"
            message = "👍 Good effort! Keep working hard."
            color = Fore.YELLOW
        elif marks >= 60:
            grade = "C"
            message = "🙂 Not bad! You can do even better next time."
            color = Fore.MAGENTA
        elif marks >= 50:
            grade = "D"
            message = "💡 You passed! Try to push yourself a little more."
            color = Fore.BLUE
        else:
            grade = "F"
            message = "💔 Don’t give up! Every failure is a step to success."
            color = Fore.RED

        
        print(color + f"\n🎯 Your Grade: {grade}")
        print(color + f"{message}\n")

except ValueError:
    print(Fore.RED + "⚠️ Invalid input! Please enter a number.")
