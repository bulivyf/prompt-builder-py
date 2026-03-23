python -m nuitka ^
  --standalone ^
  --assume-yes-for-downloads ^
  --output-dir=dist ^
  --output-filename=PromptBuilder.exe ^
  --include-package=app ^
  --include-package=fastapi ^
  --include-package=starlette ^
  --include-package=uvicorn ^
  --include-package=anyio ^
  --include-package=pydantic ^
  --include-package=jinja2 ^
  --include-data-dir=app\static=app\static ^
  --include-data-dir=app\templates=app\templates ^
  --include-data-dir=images=images ^
  --include-data-files=app_data\prompt_builder.sqlite3=app_data\prompt_builder.sqlite3 ^
  run_server.py