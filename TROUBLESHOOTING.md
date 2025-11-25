# TradingAgents - Quick Troubleshooting Guide

## Common Issues and Solutions

### 1. "api_key client option must be set" Error

**Symptoms**:
```
The api_key client option must be set either by passing api_key to the client 
or by setting the OPENAI_API_KEY environment variable
```

**Solution**:
1. Ensure you have a `.env` file in the project root
2. Check that `OPENAI_API_KEY` is set in `.env`:
   ```bash
   cat .env | grep OPENAI_API_KEY
   ```
3. If missing, add it:
   ```bash
   echo "OPENAI_API_KEY=your_key_here" >> .env
   ```
4. Run the diagnostic:
   ```bash
   .venv/bin/python diagnose.py
   ```

### 2. "ALPHA_VANTAGE_API_KEY environment variable is not set"

**Symptoms**:
```
ValueError: ALPHA_VANTAGE_API_KEY environment variable is not set.
```

**Solution**:
1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Add it to your `.env` file:
   ```bash
   echo "ALPHA_VANTAGE_API_KEY=your_key_here" >> .env
   ```
3. Alternatively, change data vendor settings in code:
   ```python
   config["data_vendors"]["fundamental_data"] = "openai"  # Use OpenAI instead
   ```

### 3. Module Not Found Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'openai'
ImportError: cannot import name 'ChatOpenAI' from 'langchain_openai'
```

**Solution**:
1. Ensure you're using the virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
2. Install/update dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify installation:
   ```bash
   python diagnose.py
   ```

### 4. "No such file or directory: .env"

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.env'
```

**Solution**:
1. Create `.env` from example:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your API keys:
   ```bash
   nano .env  # or use your favorite editor
   ```

### 5. Python Version Issues

**Symptoms**:
```
SyntaxError: invalid syntax
TypeError: unsupported operand type(s)
```

**Solution**:
1. Check Python version:
   ```bash
   python --version
   ```
2. Ensure Python 3.10+ is installed
3. Create a new virtual environment with correct version:
   ```bash
   conda create -n tradingagents python=3.11
   conda activate tradingagents
   pip install -r requirements.txt
   ```

### 6. Rate Limit Errors (Alpha Vantage)

**Symptoms**:
```
AlphaVantageRateLimitError: Alpha Vantage rate limit exceeded
```

**Solution**:
1. Wait a minute before retrying
2. Use a lower research depth (1 instead of 5)
3. Consider switching to yfinance for some data:
   ```python
   config["data_vendors"]["fundamental_data"] = "yfinance"
   ```
4. Get a premium Alpha Vantage key for higher limits

### 7. OpenAI Rate Limit or Quota Errors

**Symptoms**:
```
openai.RateLimitError: Rate limit exceeded
openai.error.QuotaExceeded: You exceeded your current quota
```

**Solution**:
1. Check your OpenAI usage dashboard
2. Add payment method if needed
3. Use cheaper models for testing:
   ```bash
   python -m cli.main --shallow-model gpt-4o-mini --deep-model o1-mini
   ```
4. Reduce research depth to 1 for initial testing

### 8. "No analysts selected" Error

**Symptoms**:
```
ValueError: Trading Agents Graph Setup Error: no analysts selected!
```

**Solution**:
1. When prompted in CLI, select at least one analyst
2. In code, pass analyst list:
   ```python
   ta = TradingAgentsGraph(
       selected_analysts=["market", "news"],  # At least one
       config=config
   )
   ```
3. For CLI non-interactive mode:
   ```bash
   python -m cli.main --ticker SPY --date 2024-01-01 --analysts market,news
   ```

### 9. Model Not Found Errors

**Symptoms**:
```
openai.error.InvalidRequestError: The model 'o4-mini' does not exist
```

**Solution**:
1. Use correct model names:
   - ✅ `o1-mini` (not `o4-mini`)
   - ✅ `gpt-4o-mini`
   - ✅ `gpt-4o`
2. Check available models in `cli/utils.py`
3. Verify model access in your OpenAI account

### 10. Import Errors After Update

**Symptoms**:
```
ImportError: cannot import name 'X' from 'Y'
AttributeError: module 'X' has no attribute 'Y'
```

**Solution**:
1. Clean Python cache:
   ```bash
   find . -type d -name "__pycache__" -exec rm -r {} +
   find . -type f -name "*.pyc" -delete
   ```
2. Reinstall dependencies:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```
3. Restart Python interpreter/kernel

## Quick Diagnostics Commands

```bash
# Full diagnostic check
python diagnose.py

# Check environment variables
cat .env

# Test OpenAI connection
python -c "from openai import OpenAI; import os; from dotenv import load_dotenv; load_dotenv(); print(OpenAI(api_key=os.getenv('OPENAI_API_KEY')).models.list().data[0].id)"

# Check Python version
python --version

# List installed packages
pip list | grep -E "openai|langchain|langgraph"

# Verify virtual environment is active
which python

# Check if .env is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

## Getting Help

1. **Run diagnostics first**:
   ```bash
   python diagnose.py
   ```

2. **Check logs**: Look in terminal output for specific error messages

3. **Review documentation**:
   - `README.md` - General usage
   - `BUGFIXES.md` - Recent fixes
   - This file - Troubleshooting

4. **Community support**:
   - GitHub Issues: Report bugs
   - Discord: Real-time help
   - Documentation: Examples and guides

## Prevention Tips

1. **Always activate virtual environment** before running commands
2. **Keep dependencies updated**: `pip install -U -r requirements.txt`
3. **Secure your API keys**: Never commit `.env` to git
4. **Test with cheap models first** before production runs
5. **Monitor API usage** to avoid unexpected charges
6. **Use dry-run mode** to verify configuration:
   ```bash
   python -m cli.main --ticker SPY --date 2024-01-01 --analysts market --depth 1 --provider openai --shallow-model gpt-4o-mini --deep-model o1-mini --dry-run
   ```

## Still Having Issues?

If none of these solutions work:

1. Create a fresh virtual environment
2. Clone the repository again
3. Run the full setup from scratch:
   ```bash
   git clone https://github.com/TauricResearch/TradingAgents.git
   cd TradingAgents
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your keys
   python diagnose.py
   ```

4. If still failing, open a GitHub issue with:
   - Output of `python diagnose.py`
   - Python version
   - Operating system
   - Full error traceback
