# NaNsense

Datathon 2025 RAG Challenge

### Setup

Create the virtual environment:
```
python3 -m venv .venv
```

Activate the virtual environment:
```
source .venv/bin/activate
```

Install the dependencies:
```
pip install -r requirements.txt
```

### Run

Make sure the environment variable `OPENAI_API_KEY` is set.

```
python src/interface.py
```

As our vector database is quite large (~22GB), we subsampled it to smaller database which we include in this repository.

### Demo

<video width="640" height="360" controls>
  <source src="screen_recording.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>