#!/usr/bin/env python3
"""
Diagnostic script to check TradingAgents environment setup and configuration.
Run this script to verify that all dependencies and API keys are properly configured.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("  ⚠️  Warning: Python 3.10+ is recommended")
        return False
    return True

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_path = Path(".env")
    if not env_path.exists():
        print("✗ .env file not found")
        print("  Run: cp .env.example .env")
        print("  Then edit .env with your API keys")
        return False
    
    print("✓ .env file exists")
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key (required for most operations)",
        "ALPHA_VANTAGE_API_KEY": "Alpha Vantage API key (required for fundamental/news data)",
    }
    
    optional_vars = {
        "GOOGLE_API_KEY": "Google Generative AI API key (optional, for Google models)",
        "ANTHROPIC_API_KEY": "Anthropic API key (optional, for Claude models)",
        "TRADINGAGENTS_RESULTS_DIR": "Custom results directory (optional)",
        "TRADINGAGENTS_DATA_DIR": "Custom data directory (optional)",
    }
    
    all_good = True
    
    print("\nRequired API Keys:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value and len(value) > 10:
            masked = value[:8] + "..." + value[-4:]
            print(f"  ✓ {var}: {masked}")
        else:
            print(f"  ✗ {var}: Not set or invalid")
            print(f"    {description}")
            all_good = False
    
    print("\nOptional API Keys:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if "KEY" in var:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else value
                print(f"  ✓ {var}: {masked}")
            else:
                print(f"  ✓ {var}: {value}")
        else:
            print(f"  ○ {var}: Not set")
            print(f"    {description}")
    
    return all_good

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        ("openai", "OpenAI Python SDK"),
        ("langchain_openai", "LangChain OpenAI integration"),
        ("langchain_anthropic", "LangChain Anthropic integration"),
        ("langchain_google_genai", "LangChain Google integration"),
        ("langgraph", "LangGraph framework"),
        ("chromadb", "Vector database for memory"),
        ("yfinance", "Yahoo Finance data"),
        ("pandas", "Data manipulation"),
        ("rich", "CLI formatting"),
        ("questionary", "Interactive CLI prompts"),
        ("fastapi", "Web API framework"),
        ("dotenv", "Environment variable loading"),
    ]
    
    print("\nDependencies:")
    all_installed = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}: Installed")
        except ImportError:
            print(f"  ✗ {package}: Not installed")
            print(f"    {description}")
            all_installed = False
    
    if not all_installed:
        print("\n  Run: pip install -r requirements.txt")
    
    return all_installed

def check_openai_connection():
    """Test OpenAI API connection."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n✗ Cannot test OpenAI connection: API key not set")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test with a minimal completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' only"}],
            max_tokens=5
        )
        
        print("\n✓ OpenAI API connection: Working")
        print(f"  Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"\n✗ OpenAI API connection failed: {e}")
        return False

def check_directory_structure():
    """Check if required directories exist."""
    required_dirs = [
        "tradingagents",
        "tradingagents/agents",
        "tradingagents/dataflows",
        "tradingagents/graph",
        "cli",
        "web",
    ]
    
    print("\nDirectory Structure:")
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all diagnostic checks."""
    print("=" * 70)
    print("TradingAgents Environment Diagnostic")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 70)
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"✗ Error during check: {e}")
            results[name] = False
    
    # Optional: Test OpenAI connection
    print("\nOptional Checks:")
    print("-" * 70)
    print("OpenAI API Connection Test:")
    try:
        check_openai_connection()
    except Exception as e:
        print(f"✗ Could not test OpenAI connection: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✓ All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("  - Run the CLI: python -m cli.main")
        print("  - Or use in code: see README.md for examples")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Set up API keys: cp .env.example .env && edit .env")
        print("  3. Ensure Python 3.10+ is installed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
