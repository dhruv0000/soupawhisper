SHELL := /bin/bash

APP_NAME := soupawhisper
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CONFIG_DIR := $(HOME)/.config/$(APP_NAME)
CONFIG_FILE := $(CONFIG_DIR)/config.ini
SERVICE_DIR := $(HOME)/.config/systemd/user
SERVICE_FILE := $(SERVICE_DIR)/$(APP_NAME).service
PYTHON := poetry run python

.DEFAULT_GOAL := help

.PHONY: help install deps config clean run debug-keys version service-install start stop restart status logs service-uninstall

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Run the interactive installer script
	@chmod +x ./install.sh
	@./install.sh

deps: ## Install Python dependencies with Poetry
	@poetry install

config: ## Create ~/.config/soupawhisper/config.ini if missing
	@mkdir -p "$(CONFIG_DIR)"
	@if [ -f "$(CONFIG_FILE)" ]; then \
		echo "Config already exists at $(CONFIG_FILE)"; \
	else \
		cp "$(PROJECT_DIR)/config.example.ini" "$(CONFIG_FILE)"; \
		echo "Created config at $(CONFIG_FILE)"; \
	fi

clean: ## Remove generated local artifacts
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.wav' -o -name '*.log' \) -delete

run: ## Run SoupaWhisper manually
	@$(PYTHON) dictate.py

debug-keys: ## Print detected global key events
	@$(PYTHON) dictate.py --debug-keys

version: ## Print the app version
	@$(PYTHON) dictate.py --version

service-install: ## Install or refresh the user systemd service
	@venv_path="$$(poetry env info --path 2>/dev/null || true)"; \
	if [ -z "$$venv_path" ]; then \
		echo "Poetry environment not found. Run 'make deps' first."; \
		exit 1; \
	fi; \
	mkdir -p "$(SERVICE_DIR)"; \
	printf '%s\n' \
		'[Unit]' \
		'Description=SoupaWhisper Voice Dictation' \
		'After=graphical-session.target' \
		'' \
		'[Service]' \
		'Type=simple' \
		'WorkingDirectory=$(PROJECT_DIR)' \
		"ExecStart=$$venv_path/bin/python -u $(PROJECT_DIR)/dictate.py" \
		'Restart=on-failure' \
		'RestartSec=5' \
		'' \
		'[Install]' \
		'WantedBy=default.target' \
		> "$(SERVICE_FILE)"; \
	systemctl --user daemon-reload; \
	systemctl --user enable "$(APP_NAME)"; \
	echo "Installed $(SERVICE_FILE)"

start: ## Start the user systemd service
	@systemctl --user start "$(APP_NAME)"

stop: ## Stop the user systemd service
	@systemctl --user stop "$(APP_NAME)"

restart: ## Restart the user systemd service
	@systemctl --user restart "$(APP_NAME)"

status: ## Show service status
	@systemctl --user status "$(APP_NAME)"

logs: ## Follow service logs
	@journalctl --user -u "$(APP_NAME)" -f

service-uninstall: ## Disable and remove the user systemd service file
	@if systemctl --user list-unit-files "$(APP_NAME).service" >/dev/null 2>&1; then \
		systemctl --user disable --now "$(APP_NAME)" || true; \
	fi
	@rm -f "$(SERVICE_FILE)"
	@systemctl --user daemon-reload
	@echo "Removed $(SERVICE_FILE)"
