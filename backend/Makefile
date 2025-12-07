# YieldGuard Lite - Backend Makefile

.PHONY: install lint format check fix test run clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run linter (check only)
lint:
	ruff check .

# Run formatter (check only)
format-check:
	ruff format --check .

# Format code (apply changes)
format:
	ruff format .

# Fix lint issues (apply changes)
fix:
	ruff check --fix .

# Check everything (lint + format check)
check:
	@echo "Running linter..."
	ruff check .
	@echo "\nRunning format check..."
	ruff format --check .
	@echo "\n✅ All checks passed!"

# Fix everything (lint fix + format)
fix-all:
	@echo "Fixing lint issues..."
	ruff check --fix .
	@echo "\nFormatting code..."
	ruff format .
	@echo "\n✅ All fixes applied!"

# Run tests
test:
	python -m pytest test_agent.py -v

# Run development server
run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Help
help:
	@echo "YieldGuard Lite Backend Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make lint         - Run linter (check only)"
	@echo "  make format       - Format code"
	@echo "  make format-check - Check formatting"
	@echo "  make check        - Run all checks"
	@echo "  make fix          - Fix lint issues"
	@echo "  make fix-all      - Fix everything (lint + format)"
	@echo "  make test         - Run tests"
	@echo "  make run          - Run dev server"
	@echo "  make clean        - Clean cache files"
