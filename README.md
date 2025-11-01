# Enhanced YouTube Music Bot 🎵

A professional Discord music bot focused on YouTube content with advanced features and beautiful UI.

## 🚀 Features

### Core Music Features
- 🎵 **YouTube-only focus** - Music, podcasts, and long-form content
- 📋 **Smart queue management** with anti-spam protection
- 🔍 **Advanced search** with interactive selection
- 🎛️ **14 audio effects** including bass boost, nightcore, reverb
- 📱 **Interactive Discord UI** with buttons and controls
- ⏳ **Real-time progress tracking** with beautiful progress bars

### Advanced Features
- 🔊 **Volume control** (0-200%)
- 🔁 **Loop mode** for current track
- 🔀 **Queue shuffling** and management
- 📜 **Play history** tracking
- 🎯 **Rate limiting** to prevent spam
- 🤖 **Auto-disconnect** when idle
- 💾 **Persistent queue** with database storage

### User Experience
- ✨ **Beautiful embeds** with thumbnails and metadata
- 🎮 **Interactive controls** via Discord buttons
- 📊 **Detailed track information** including views and uploader
- 🚀 **Fast response times** with optimized code
- 🛡️ **Error handling** with user-friendly messages

## 📋 Requirements

- Python 3.8+
- Discord Bot Token
- FFmpeg installed on system

## 🛠️ Installation

1. **Clone or download the files**
```bash
# Download musicbot.py and requirements.txt
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg**
- **Windows**: Download from https://ffmpeg.org/download.html
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your Discord bot token
```

5. **Run the bot**
```bash
python musicbot.py
```

## 🔧 Configuration

### Discord Bot Setup
1. Go to https://discord.com/developers/applications
2. Create a new application
3. Go to "Bot" section and create a bot
4. Copy the token to your `.env` file
5. Enable "Message Content Intent" in bot settings

### Bot Permissions
The bot needs these permissions:
- Connect and Speak in voice channels
- Send Messages and Embed Links
- Use Slash Commands
- View Channels

### Environment Variables
```env
DISCORD_TOKEN=your_bot_token_here
APP_ID=your_application_id
DATABASE_URL=sqlite:///musicbot.db
MAX_QUEUE_SIZE=100
MAX_USER_QUEUE=10
MAX_TRACK_LENGTH=7200
IDLE_TIMEOUT=300
HISTORY_DAYS=7
```

## 📱 Commands

### Basic Commands
- `/join` - Join your voice channel
- `/leave` - Leave voice channel and clear queue
- `/play <query>` - Play music from YouTube (URL or search)
- `/search <query>` - Search YouTube and choose from results
- `/pause` - Pause playback
- `/resume` - Resume playback
- `/skip` - Skip current track
- `/stop` - Stop playback and clear queue

### Queue Management
- `/queue` - Show current queue with progress
- `/shuffle` - Shuffle the queue randomly
- `/clear` - Clear entire queue
- `/loop` - Toggle loop mode for current track
- `/nowplaying` - Show detailed current track info

### Audio Controls
- `/volume <0-200>` - Set playback volume
- `/effects <effect>` - Toggle audio effects:
  - Bass Boost, Nightcore, Vaporwave
  - Treble Boost, Vocal Boost, Karaoke
  - Vibrato, Tremolo, Chorus
  - Reverb, Echo, Distortion
  - Mono, Stereo Enhance

### Information
- `/history` - Show recently played tracks
- `/help` - Show all available commands

## 🎛️ Audio Effects

The bot includes 14 professional audio effects:

- **Bass Boost** - Enhanced low frequencies
- **Nightcore** - Higher pitch and tempo
- **Vaporwave** - Slowed down retro sound
- **Treble Boost** - Enhanced high frequencies
- **Vocal Boost** - Emphasize vocal frequencies
- **Karaoke** - Remove center vocals
- **Vibrato** - Pitch modulation effect
- **Tremolo** - Volume modulation effect
- **Chorus** - Rich, layered sound
- **Reverb** - Spacious echo effect
- **Echo** - Distinct echo repeats
- **Distortion** - Aggressive sound distortion
- **Mono** - Convert to mono audio
- **Stereo Enhance** - Widen stereo field

## 🏗️ Architecture

### Core Components
- **YouTubeExtractor** - Handles all YouTube interactions
- **AudioEffectsProcessor** - Manages FFmpeg audio processing
- **DatabaseManager** - SQLite database operations
- **MusicControlView** - Interactive Discord UI components

### Database Schema
- **queue** - Current queue storage
- **history** - Play history tracking
- **server_configs** - Per-server settings
- **user_stats** - User statistics

### Performance Features
- Async/await throughout for non-blocking operations
- Smart caching to reduce API calls
- Rate limiting to prevent abuse
- Background cleanup tasks
- Optimized database queries

## 🔒 Security Features

- **Rate limiting** - Prevents command spam
- **Input validation** - Sanitizes all user inputs
- **Error handling** - Graceful error recovery
- **Resource cleanup** - Prevents memory leaks
- **Safe disconnection** - Proper cleanup on shutdown

## 🐛 Troubleshooting

### Common Issues

**Bot doesn't respond to commands:**
- Check if bot has proper permissions
- Ensure slash commands are synced (restart bot)
- Verify bot token is correct

**Audio doesn't play:**
- Install FFmpeg and ensure it's in PATH
- Check voice channel permissions
- Verify YouTube URL is accessible

**Database errors:**
- Ensure write permissions in bot directory
- Check if SQLite file is corrupted
- Restart bot to recreate database

### Logging
The bot creates detailed logs in `musicbot.log` for debugging.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

- Check the logs in `musicbot.log`
- Ensure all requirements are installed
- Verify Discord bot permissions
- Test with simple YouTube URLs first

---

**Note**: This bot is designed for educational and personal use. Please respect YouTube's terms of service and copyright laws.