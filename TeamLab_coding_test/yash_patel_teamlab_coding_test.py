# 1. Sum of odd cubes

sum_of_cubes = 0

for num in range(1, 78):
    if num % 2 != 0:  # Check if number is odd
        sum_of_cubes += num ** 3

print("The sum of the cubes of odd numbers from 1 to 77 is:", sum_of_cubes)


# 2. Anomalous Fibonacci series

def anomalous_fibonacci(n):
    sequence = [1, 0, 5]  # Starting sequence
    if n <= 3:
        return sequence[n-1]
    else:
        for i in range(3, n):
            next_number = sequence[i-1] + sequence[i-3]
            sequence.append(next_number)
        return sequence[n-1]

# Calculate the 42nd integer
n = 42
result = anomalous_fibonacci(n)
print(f"The {n}-th integer in the anomalous Fibonacci series is: {result}")

# 3. 5-letter words

def count_strings():
    count = 0

    characters = "ABCDEFGJKQTVWXYZ"

    for a in characters:
        for b in characters:
            for c in characters:
                for d in characters:
                    for e in characters:
                        # Check conditions
                        if "A" in [a, b, c, d, e] and b != "A" and "E" in [a, b, c, d, e] and d != "E" and "T" in [a, b, c, d, e] and b != "T":
                            count += 1

    return count

# Call the function and print the result
num_strings = count_strings()
print("Number of strings that satisfy the conditions:", num_strings)

# 4. Stamp, stamp, stamp

def calculate_values():
    stamp_counts = [30, 40, 30]
    stamp_values = [205, 82, 30]

    total_values = set()
    for i in range(len(stamp_counts)):
        new_values = set()
        for j in range(1, stamp_counts[i] + 1):
            for value in total_values:
                new_values.add(value + stamp_values[i] * j)
            new_values.add(stamp_values[i] * j)
        total_values |= new_values

    return len(total_values)

num_values = calculate_values()
print("Number of values that can be formed:", num_values)

# 5. Too many boxes

boxes = range(500, 0, -1)  # List of box weights from 500kg to 1kg
capacity = 5000  # Maximum capacity of each truck
trucks = 0  # Number of trucks used
remaining_capacity = capacity  # Remaining capacity of the current truck

for box in boxes:
    if box <= remaining_capacity:
        remaining_capacity -= box
    else:
        trucks += 1
        remaining_capacity = capacity - box

trucks += 1  # Account for the last truck

print("Number of trucks needed:", trucks)

# 6. Divisor Sort

def smallest_divisor(n):
    """Returns the smallest divisor greater than 1 for a given number."""
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i
    return n

def compare_integers(a):
    """Comparison function to order the integers based on the given rules."""
    divisor_a = smallest_divisor(a)

    return (-divisor_a, -a)  # Compare in descending order of the smallest divisors and the numbers themselves

# Step 1: Create a list of integers from 2 to 1,000,000
integers = list(range(2, 1000001))

# Step 2: Sort the list using the comparison function
integers.sort(key=compare_integers)

# Step 4: Return the integer at index 230,000
result = integers[210000]

print("The 230,001st integer in the ordered list is:", result)


# 7. Numeri Romani

def roman_to_int(roman):
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
    }
    total = 0
    prev_value = 0
    
    for char in reversed(roman):
        value = roman_values[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total

def int_to_roman(n):
    roman_values = {
        1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 100: 'C',
        90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX', 5: 'V', 4: 'IV', 1: 'I'
    }
    roman = ''
    
    for value, symbol in roman_values.items():
        while n >= value:
            roman += symbol
            n -= value
    
    return roman

total_sum = 0

for num in range(1, 1001):
    roman_num = int_to_roman(num)
    if len(roman_num) == 9:
        total_sum += num

print(total_sum)

# 8. Spiral Letters

def generate_spiral_alphabet(size):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    spiral = [['' for _ in range(size)] for _ in range(size)]

    # Define the boundaries of the spiral
    top = 0
    bottom = size - 1
    left = 0
    right = size - 1

    direction = 0  # 0: moving right, 1: moving down, 2: moving left, 3: moving up
    letter_index = 0  # Current index of the alphabet

    while top <= bottom and left <= right:
        if direction == 0:  # Moving right
            for i in range(left, right + 1):
                spiral[top][i] = alphabet[letter_index]
                letter_index = (letter_index + 1) % 26
            top += 1
        elif direction == 1:  # Moving down
            for i in range(top, bottom + 1):
                spiral[i][right] = alphabet[letter_index]
                letter_index = (letter_index + 1) % 26
            right -= 1
        elif direction == 2:  # Moving left
            for i in range(right, left - 1, -1):
                spiral[bottom][i] = alphabet[letter_index]
                letter_index = (letter_index + 1) % 26
            bottom -= 1
        elif direction == 3:  # Moving up
            for i in range(bottom, top - 1, -1):
                spiral[i][left] = alphabet[letter_index]
                letter_index = (letter_index + 1) % 26
            left += 1

        direction = (direction + 1) % 4

    diagonal_letters = [spiral[i][i] for i in range(size)]  # Extract the diagonal letters
    return ''.join(diagonal_letters)


board_size = 40
diagonal_letters = generate_spiral_alphabet(board_size)
print(diagonal_letters)

# 9. Knight Trip

def valid_moves(x, y):
    moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
    valid = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 50 and 0 <= ny < 50:
            valid.append((nx, ny))
    return valid

def knight_moves(start_x, start_y, num_moves):
    positions = {(start_x, start_y)}
    for _ in range(num_moves):
        new_positions = set()
        for x, y in positions:
            moves = valid_moves(x, y)
            new_positions.update(moves)
        positions = new_positions
    return len(positions)

start_x = 0  # top left square x-coordinate
start_y = 0  # top left square y-coordinate
num_moves = 22

num_squares = knight_moves(start_x, start_y, num_moves)
print(f"The knight can end on {num_squares} different squares after {num_moves} moves.")

# 10. Stalin Sort

def operation_s(numbers):
    result = []
    biggest = None

    for num in numbers:
        if biggest is None or num >= biggest:
            biggest = num
            result.append(num)

    return result


def count_lists_with_two_numbers():
    count = 0
    for n1 in range(10):
        for n2 in range(10):
            for n3 in range(10):
                for n4 in range(10):
                    for n5 in range(10):
                        for n6 in range(10):
                            for n7 in range(10):
                                numbers = [n1, n2, n3, n4, n5, n6, n7]
                                result = operation_s(numbers)
                                if len(result) == 2:
                                    count += 1

    return count

# Calculate the number of lists
num_lists = count_lists_with_two_numbers()
print(num_lists)

