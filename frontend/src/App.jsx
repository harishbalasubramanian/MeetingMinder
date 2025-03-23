import { useState } from 'react';
import axios from 'axios';

export default function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showFullTranscript, setShowFullTranscript] = useState(false);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setIsLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await axios.post(
        'http://localhost:8000/analyze/',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      setResults(response.data);
    } catch (err) {
      setError('Failed to process. Try again');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const calculateOverallEmotion = (windows) => {
    const emotionCounts = windows.reduce((acc, window) => {
      const emotion = window.combined_sentiment || window.audio_emotion;
      if (emotion) {
        const key = emotion.toLowerCase();
        acc[key] = (acc[key] || 0) + 1;
      }
      return acc;
    }, {});

    if (Object.keys(emotionCounts).length === 0) return 'Neutral';
    
    const maxEntry = Object.entries(emotionCounts).reduce(
      (max, [emotion, count]) => count > max[1] ? [emotion, count] : max,
      ['neutral', 0]
    );

    return maxEntry[0].charAt(0).toUpperCase() + maxEntry[0].slice(1);
  };

  const getSentimentColor = (sentiment) => {
    if (!sentiment) return '#9CA3AF';
    switch (sentiment.toLowerCase()) {
      case 'positive': return '#34D399';
      case 'negative': return '#F87171';
      default: return '#9CA3AF';
    }
  };

  const getSentimentBadge = (sentiment) => {
    if (!sentiment) return 'bg-gray-100 text-gray-800';
    switch (sentiment.toLowerCase()) {
      case 'positive': return 'bg-green-100 text-green-800';
      case 'negative': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
              MeetingMinder
            </span>
          </h1>
          <p className="text-xl text-gray-600">
            Transform meetings into actionable insights
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-xl p-8 mb-8">
          <form onSubmit={handleUpload} className="space-y-6">
            <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-8 transition-all hover:border-blue-500">
              <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
                accept=".mp3,.wav,.mp4,.mov"
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700 transition-colors"
              >
                {file ? file.name : 'Choose Meeting Recording'}
              </label>
              <p className="mt-4 text-sm text-gray-600">
                MP4 or MOV or MP3 or WAV files only Max 50MB
              </p>
            </div>

            <button
              type="submit"
              disabled={isLoading || !file}
              className={`w-full py-3 px-6 text-white font-medium rounded-md transition-colors ${isLoading || !file ? 'bg-gray-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700'}`}
            >
              {isLoading ? "Processing..." : "Analyze Meeting"}
            </button>

            {error && (
              <div className="p-4 bg-red-50 text-red-700 rounded-md">
                {error}
              </div>
            )}
          </form>
        </div>

        {results && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-xl p-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Meeting Summary
              </h2>
              <p className="text-gray-700 leading-relaxed">
                {results.summary}
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-xl p-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Action Items
              </h2>
              <ul className="space-y-4">
                {results.action_items.map((item, index) => (
                  <li
                    key={index}
                    className="flex items-start space-x-3 bg-blue-50 p-4 rounded-md"
                  >
                    <div className="flex-shrink-0 mt-1">
                      <div className="w-3 h-3 bg-blue-600 rounded-full" />
                    </div>
                    <span className="text-gray-700">{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-white rounded-lg shadow-xl p-8">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-semibold text-gray-900">
                  Full Transcript
                </h2>
                <button
                  onClick={() => setShowFullTranscript(!showFullTranscript)}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  {showFullTranscript ? 'Show Less' : 'Read More'}
                </button>
              </div>
              <p className="text-gray-700 leading-relaxed">
                {showFullTranscript
                  ? results.transcript
                  : `${results.transcript.substring(0, 300)}...`}
              </p>
            </div>

            {results.video_metadata && (
              <div className="bg-white rounded-lg shadow-xl p-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                  Video Details
                </h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-gray-600">Duration:</span>
                    <span className="ml-2 text-gray-800">
                      {Math.round(results.video_metadata.duration)} seconds
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">Resolution:</span>
                    <span className="ml-2 text-gray-800">
                      {results.video_metadata.width}x{results.video_metadata.height}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {results.sentiment_windows && results.sentiment_windows.length > 0 && (
              <div className="bg-white rounded-lg shadow-xl p-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                  Sentiment Analysis
                </h2>

                <div className="mb-8 p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <span className="text-lg font-medium text-gray-700">
                      Overall Tone:
                    </span>
                    <span className={`px-3 py-1 rounded-full ${getSentimentBadge(calculateOverallEmotion(results.sentiment_windows))}`}>
                      {calculateOverallEmotion(results.sentiment_windows)}
                    </span>
                  </div>
                </div>

                <div className="mb-8">
                  <h3 className="text-lg font-medium text-gray-700 mb-2">
                    Sentiment Timeline
                  </h3>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    {results.sentiment_windows.map((window, index) => {
                      const emotion = window.combined_sentiment || window.audio_emotion;
                      return (
                        <div
                          key={index}
                          className="inline-block h-full"
                          style={{
                            width: `${100 / results.sentiment_windows.length}%`,
                            backgroundColor: getSentimentColor(emotion)
                          }}
                          title={`${window.start_time.toFixed(1)}-${window.end_time.toFixed(1)}s: ${emotion || 'No data'}`}
                        />
                      );
                    })}
                  </div>
                </div>

                <div className="space-y-4">
                  {results.sentiment_windows.map((window, index) => {
                    const emotion = window.combined_sentiment || window.audio_emotion;
                    return (
                      <div key={index} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                        <div className="flex justify-between items-start mb-2">
                          <span className="font-medium text-gray-700">
                            {window.start_time.toFixed(1)}-{window.end_time.toFixed(1)}s
                          </span>
                          <span className={`px-2 py-1 rounded ${getSentimentBadge(emotion)}`}>
                            {emotion || 'Unknown'}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-gray-600">Audio Emotion:</p>
                            <p className="font-medium">{window.audio_emotion || 'N/A'}</p>
                          </div>
                          <div>
                            <p className="text-gray-600">Visual Emotion:</p>
                            <p className="font-medium">{window.visual_emotion || 'N/A'}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}