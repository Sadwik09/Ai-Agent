import json
import time
from datetime import datetime
from logging_config import project_status, master_logger

def format_status():
    status = project_status.get_status()
    formatted = {
        'timestamp': datetime.now().isoformat(),
        'projects': {}
    }
    
    for project, data in status.items():
        formatted['projects'][project] = {
            'status': data['status'],
            'last_update': data['last_update'].isoformat() if data['last_update'] else None,
            'error_count': len(data['errors']),
            'warning_count': len(data['warnings']),
            'latest_error': data['errors'][-1] if data['errors'] else None,
            'latest_warning': data['warnings'][-1] if data['warnings'] else None
        }
    
    return formatted

def save_status():
    status = format_status()
    with open('logs/status.json', 'w') as f:
        json.dump(status, f, indent=2)

def monitor():
    master_logger.info("Starting status monitoring...")
    while True:
        try:
            save_status()
            status = format_status()
            
            # Print current status
            print("\nCurrent Project Status:")
            print("=" * 50)
            for project, data in status['projects'].items():
                print(f"\n{project.upper()}:")
                print(f"Status: {data['status']}")
                print(f"Last Update: {data['last_update']}")
                print(f"Errors: {data['error_count']}")
                print(f"Warnings: {data['warning_count']}")
                if data['latest_error']:
                    print(f"Latest Error: {data['latest_error']}")
                if data['latest_warning']:
                    print(f"Latest Warning: {data['latest_warning']}")
            print("\n" + "=" * 50)
            
            time.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            master_logger.error(f"Error in status monitoring: {str(e)}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    monitor() 