import threading
import time
import concurrent.futures

class ParkingGate:
    def __init__(self, total_spaces):
        self.total_spaces = total_spaces
        self.available_spaces = total_spaces
        self.lock = threading.Lock()  # Lock to ensure thread safety

    def enter(self):
        with self.lock:
            if self.available_spaces > 0:
                self.available_spaces -= 1
                print(f"Available spaces: {self.available_spaces}/{self.total_spaces}")
            else:
                print("Parking is full.")

    def exit(self):
        with self.lock:
            if self.available_spaces < self.total_spaces:
                self.available_spaces += 1
                print(f"Available spaces: {self.available_spaces}/{self.total_spaces}")
            else:
                print("Parking is empty.")

def simulation(parking_gate):
    while True:
        action = input("Enter 'e' to simulate car entry, 'x' to simulate car exit, or 'q' to quit the simulation: ")
        if action == 'e':
            parking_gate.enter()
        elif action == 'x':
            parking_gate.exit()
        elif action == 'q':
            break
        else:
            print("Invalid input. Please enter 'e', 'x', or 'q'.")

if __name__ == "__main__":
    total_spaces = 10  # Total parking spaces
    gate = ParkingGate(total_spaces)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.submit(simulation, gate)