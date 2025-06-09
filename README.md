# SpotiFind
Inspired by Shazam :D
SpotiFind is a music similarity search application I made. It can parse youtube playlist/videos and add to your database, save songs locally to your device, and search for song through audio like Shazam.

## Backend

### Frameworks and Tools
- **Framework**: Flask
- **Database**: PostgreSQL with pgvector for efficient vector similarity search
- **Audio Processing**: OpenL3 for generating audio embeddings
- **YouTube Integration**: YouTube-DLP API for downloading and processing YouTube videos

### Audio Preprocessing and Processing pipeline

#### Preprocessing
1. **Format Conversion**:
   - Audio files are converted to `.wav` format using `ffmpeg`.

2. **Resampling**:
   - Files are resampled to a standard sample rate.

3. **Mono Conversion**:
   - Stereo audio is converted to mono for consistency.

4. **Trimming**:
   - Audio is trimmed to a fixed duration (60 seconds).

#### Processing
1. **Softening sudden changes in audio volume**:
   - Cut out noise that is too loud or quiet (The outliers).

2. **"Flatten" out audio**:
   - Quiet parts of the audio are enhanced relative to loud parts using upward expansion.

3. **Feature Extraction**:
   - OpenL3 generates embeddings that capture audio characteristics for similarity matching.

4. **Vector Storage**:
   - Extracted embeddings are stored in PostgreSQL with pgvector for Manhattan similarity search.

## Frontend

- **Framework**: React
- **UI**: Tailwind CSS for styling
- **Features**:
  - Upload audio files for analysis
  - Add YouTube videos or playlists
  - View results of similar songs
  - Playback uploaded recordings
