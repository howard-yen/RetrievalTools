from setuptools import setup

setup(
    name='retrievaltools-py',
    version='0.1',
    packages=['retrievaltools'],
    install_requires=[
        'wheel>=0.41.2',
        'tqdm>=4.66.1',
        'numpy>=1.26.4',
        'simple_parsing>=0.1.5',
        'datasets>=2.18.0',
        'fastapi>=0.115.2',
        'uvicorn>=0.31.0',
        'requests>=2.31.0',
        'dataclasses>=0.1.1',
        'dataclasses-json>=0.6.2',
        'datatools-py>=0.1.0',
        'transformers>=4.35.0',
        'rouge-score>=0.1.2',
        # 'sentence_transformers>=4.0.0',
        'psutil>=6.0.0',
        'beautifulsoup4>=4.13.0',
        'pymupdf>=1.25.5',
    ],
    author='Howard Yen',
    description='Library and scripts for common retrieval utilities (generating dense embeddings, faiss index, web search, etc.)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/howard-yen/RetrievalTools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'serve_retriever=retrievaltools.serve_retriever:main',
            'generate_embeddings=retrievaltools.generate_passage_embeddings:main',
            'passage_retrieval=retrievaltools.passage_retrieval:main',
        ]
    },
    python_requires='>=3.7',
)