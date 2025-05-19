
import requests
import time
import threading
from datetime import datetime
from typing import List, Dict, Any

class EndpointMonitor:
    def __init__(self, endpoints: List[str], check_interval: int = 10):
        self.endpoints = endpoints
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.available_endpoints = {}  # Store status of endpoints
        self.lock = threading.Lock()  # Thread safety for shared data

    def check_endpoint(self, url: str):
        try:
            response = requests.get(f"{url}/v1/models", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Status code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, str(e)

    def monitor_loop(self):
        while self.running:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nCheck performed at: {current_time}")
            print("Available endpoints:")
            
            available_count = 0
            
            # Update available endpoints dict
            with self.lock:
                self.available_endpoints = {}
                for endpoint in self.endpoints:
                    is_alive, result = self.check_endpoint(endpoint)
                    status = "✅ AVAILABLE" if is_alive else "❌ DOWN"
                    
                    if is_alive:
                        available_count += 1
                        self.available_endpoints[endpoint] = result
                        print(f"{status} - {endpoint}")
                    else:
                        print(f"{status} - {endpoint} - Error: {result}")
            
            print(f"\nSummary: {available_count}/{len(self.endpoints)} endpoints available")
            print("-" * 50)
            
            # Wait for check_interval seconds before the next check
            time.sleep(self.check_interval)

    def start(self):
        """Start the monitoring in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.thread.start()
            print("Endpoint monitoring started in background thread")
        else:
            print("Monitoring is already running")

    def stop(self):
        """Stop the monitoring thread"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1)
            print("Endpoint monitoring stopped")
        else:
            print("Monitoring is not running")

    def get_available_endpoints(self) -> Dict[str, Any]:
        """Return a dictionary of currently available endpoints and their model data"""
        with self.lock:
            return self.available_endpoints.copy()


# Example usage in your main application:
def main():
    # List of base URLs for the 5 endpoints
# List of base URLs for the 5 endpoints
    endpoints = [
        "http://172.27.188.32:8081",
        "http://172.27.188.32:8082",
        "http://172.27.188.40:8081",
        "http://172.27.188.40:8082",
        "http://172.27.188.31:8082"
    ]
    
    # Create and start the monitor
    monitor = EndpointMonitor(endpoints, check_interval=10)
    monitor.start()
    
    # Your main application code continues here
    try:
        # Simulate your main application running
        print("Main application is running...")
        
        # Example of accessing the available endpoints from your main app
        time.sleep(12)  # Wait for first check to complete
        print("start")
        available = monitor.get_available_endpoints()
        print(f"\nFrom main thread - Available endpoints: {list(available.keys())}")
        
        # Keep main thread alive (in a real app, you'd have your actual application logic here)
        print("end")
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nShutting down application...")
        monitor.stop()
        print("Application terminated")


if __name__ == "__main__":
    main()