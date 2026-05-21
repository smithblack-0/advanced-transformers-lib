def pytest_addoption(parser):
    parser.addoption(
        "--hub",
        choices=["dev", "main"],
        default="main",
        help="Hub target for network tests: 'main' (production) or 'dev' (staging).",
    )
