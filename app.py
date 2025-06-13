#!/usr/bin/env python3
"""
Application launcher for Agentic RAG Solution
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from config import Config

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'pymongo', 'sentence-transformers', 
        'google', 'tavily', 'langgraph'
    ]

    # Map package names to their actual module names if different
    package_mapping = {
        'google-genai': 'google_genai',
        'tavily-python': 'tavily_python'
    }

    missing_packages = []

    for package in required_packages:
        module_name = package_mapping.get(package, package.replace('-', '_'))
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def validate_configuration():
    """Validate system configuration"""
    print("🔍 Validating configuration...")
    
    validation = Config.validate()
    
    if not validation["valid"]:
        print("❌ Configuration validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        print("\n💡 Please check your .env file or environment variables.")
        return False
    
    if validation["warnings"]:
        print("⚠️  Configuration warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    print("✅ Configuration validated successfully!")
    return True

def setup_environment():
    """Setup environment for the application"""
    # Create .env file if it doesn't exist
    env_file = Path('.env')
    env_template = Path('.env.template')
    
    if not env_file.exists() and env_template.exists():
        print("📝 Creating .env file from template...")
        env_template.rename(env_file)
        print("⚠️  Please edit .env file with your API keys before running the app.")
        return False
    
    return True

def run_streamlit_app(port=8501, host='localhost', debug=False):
    """Run the Streamlit application"""
    print(f"🚀 Starting Agentic RAG Application on {host}:{port}")
    
    # Streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'agentic.py',
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'true' if not debug else 'false',
        '--theme.base', Config.STREAMLIT_THEME
    ]
    
    if debug:
        cmd.extend(['--logger.level', 'debug'])
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        return False
    
    return True

def main():
    """Main application launcher"""
    parser = argparse.ArgumentParser(description='Agentic RAG Solution Launcher')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the application on')
    parser.add_argument('--host', default='localhost', help='Host to run the application on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--check-deps', action='store_true', help='Only check dependencies')
    parser.add_argument('--validate-config', action='store_true', help='Only validate configuration')
    parser.add_argument('--setup', action='store_true', help='Setup environment')
    
    args = parser.parse_args()
    
    print("🤖 Agentic RAG Solution Launcher")
    print("=" * 40)
    
    # Setup environment if requested
    if args.setup:
        if not setup_environment():
            return 1
        print("✅ Environment setup complete!")
        return 0
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    if args.check_deps:
        print("✅ All dependencies are installed!")
        return 0
    
    # Validate configuration
    if not validate_configuration():
        return 1
    
    if args.validate_config:
        print("✅ Configuration is valid!")
        return 0
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Run the application
    if not run_streamlit_app(args.port, args.host, args.debug):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
