import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:uuid/uuid.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

const String _backendBaseUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'http://127.0.0.1:8000',
);

void main() {
  runApp(const HypnoseChatApp());
}

class HypnoseChatApp extends StatelessWidget {
  const HypnoseChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    const surface = Color(0xFF0B1223);
    const accent = Color(0xFF2C6BED);
    return MaterialApp(
      title: 'Hypnose Chat',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: accent,
          brightness: Brightness.dark,
          surface: surface,
        ),
        scaffoldBackgroundColor: const Color(0xFF060B16),
        useMaterial3: true,
      ),
      home: const ChatHomePage(),
    );
  }
}

class ChatHomePage extends StatefulWidget {
  const ChatHomePage({super.key});

  @override
  State<ChatHomePage> createState() => _ChatHomePageState();
}

class _ChatHomePageState extends State<ChatHomePage> {
  final ChatApiClient _chatApi = ChatApiClient(baseUrl: _backendBaseUrl);
  final AudioPlayer _assistantAudioPlayer = AudioPlayer();
  final stt.SpeechToText _speech = stt.SpeechToText();
  final TextEditingController _inputController = TextEditingController();
  final FocusNode _inputFocusNode = FocusNode();
  final ScrollController _scrollController = ScrollController();
  final String _sessionId = const Uuid().v4();

  final List<ChatEntry> _messages = <ChatEntry>[];

  bool _sending = false;
  bool _dictating = false;
  bool _voiceChatActive = false;
  bool _voiceThinking = false;
  bool _voiceSpeaking = false;
  bool _voiceRestartAllowed = true;
  String _voiceStatus = 'Bereit';
  double _chatTextScale = 1.0;
  bool _phase4Starting = false;
  bool _phase4Active = false;
  bool _awaitingDecisionInput = false;
  int _activeTabIndex = 0;

  Timer? _voiceRestartTimer;
  Timer? _voicePartialCommitTimer;
  DateTime? _lastVoiceTurnAt;
  String? _lastVoiceTurnText;
  String _pendingVoiceTranscript = '';
  WebSocketChannel? _voiceSpeakChannel;
  WebSocketChannel? _voiceListenChannel;
  StreamSubscription<dynamic>? _voiceSpeakSubscription;
  StreamSubscription<dynamic>? _voiceListenSubscription;
  Timer? _voiceSpeakPingTimer;
  Timer? _voiceListenPingTimer;
  Completer<void>? _voiceSpeakReadyCompleter;
  bool _voiceSpeakReady = false;
  final BytesBuilder _voiceIncomingAudioBuffer = BytesBuilder(copy: false);
  String? _voiceIncomingAudioMessageId;
  final Queue<_VoiceAudioChunk> _voiceAudioQueue = Queue<_VoiceAudioChunk>();
  bool _voiceAudioQueueRunning = false;
  bool _voiceReconnectScheduled = false;

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    _voiceRestartTimer?.cancel();
    _voicePartialCommitTimer?.cancel();
    _voiceSpeakPingTimer?.cancel();
    _voiceListenPingTimer?.cancel();
    unawaited(_disconnectVoiceLoop(closeSockets: true));
    unawaited(_assistantAudioPlayer.dispose());
    unawaited(_speech.stop());
    _chatApi.dispose();
    _inputController.dispose();
    _inputFocusNode.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _sendFromInput() async {
    final text = _inputController.text.trim();
    if (text.isEmpty || _sending || _phase4Starting) return;
    if (_phase4Active && !_awaitingDecisionInput) return;
    _inputController.clear();
    setState(() {});
    await _sendMessage(text, fromVoice: false);
  }

  Future<void> _sendMessage(String text, {required bool fromVoice}) async {
    final normalized = text.trim();
    if (normalized.isEmpty) return;
    final phase4WasActive = _phase4Active;

    if (_dictating) {
      await _stopDictation();
    }

    if (_voiceChatActive) {
      await _pauseVoiceRecognitionForTurn();
    }

    _appendMessage(ChatEntry(role: 'user', text: normalized));
    setState(() {
      _sending = true;
      _voiceThinking = _voiceChatActive || fromVoice;
      _awaitingDecisionInput = false;
      if (_voiceThinking) {
        _voiceStatus = 'Warte';
      }
    });

    try {
      final result = await _chatApi.sendMessage(
        sessionId: _sessionId,
        message: normalized,
      );
      _appendMessage(ChatEntry(role: 'assistant', text: result.reply));
      if (mounted) {
        setState(() {
          _phase4Active = result.phase4Active;
          _awaitingDecisionInput = result.awaitsUserInput;
        });
      }

      final shouldSpeak =
          _voiceChatActive ||
          fromVoice ||
          phase4WasActive ||
          result.phase4Active;
      if (shouldSpeak) {
        if (_voiceChatActive && _voiceSpeakReady) {
          await _sendSpeakTextOverWs(result.reply, emitText: false);
        } else {
          await _speakAssistantReply(result.reply);
        }
      }
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Fehler beim Senden: $error')));
      setState(() {
        _voiceStatus = _voiceChatActive ? 'HÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¶rt zu' : 'Bereit';
      });
    } finally {
      if (mounted) {
        setState(() {
          _sending = false;
          _voiceThinking = false;
          if (_voiceChatActive && !_voiceSpeaking) {
            _voiceStatus = 'HÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¶rt zu';
          }
        });
      }
      if (mounted && _voiceChatActive && !_voiceSpeaking) {
        await _startVoiceChatListening();
      }
      if (mounted) {
        await _syncDictationGate();
      }
    }
  }

  Future<void> _speakAssistantReply(String text) async {
    final clean = text.trim();
    if (clean.isEmpty) return;
    if (mounted) {
      setState(() {
        _voiceSpeaking = true;
        _voiceStatus = 'Spricht';
      });
      await _syncDictationGate();
    }

    try {
      final audioBytes = await _chatApi.requestTtsAudio(
        text: clean,
        context: 'session_phase_4',
        phase: 4,
      );
      await _assistantAudioPlayer.stop();
      await _assistantAudioPlayer.play(BytesSource(audioBytes), volume: 1.0);
      await _assistantAudioPlayer.onPlayerComplete.first.timeout(
        const Duration(minutes: 5),
      );
      if (mounted) {
        setState(() {
          _voiceSpeaking = false;
          _voiceStatus = _voiceChatActive ? 'HÃƒÂ¶rt zu' : 'Bereit';
        });
        await _syncDictationGate();
      }
      return;
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _voiceSpeaking = false;
        _voiceStatus = _voiceChatActive ? 'HÃƒÂ¶rt zu' : 'Bereit';
      });
      await _syncDictationGate();
      throw Exception('TTS fehlgeschlagen: $error');
    }
  }

  void _appendMessage(ChatEntry message) {
    setState(() => _messages.add(message));
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 180),
        curve: Curves.easeOutCubic,
      );
    });
  }

  bool _dictationDecisionWindowOpen() {
    return _activeTabIndex == 0 &&
        _phase4Active &&
        _awaitingDecisionInput &&
        !_sending &&
        !_voiceChatActive &&
        !_voiceThinking &&
        !_voiceSpeaking &&
        !_phase4Starting;
  }

  Future<void> _syncDictationGate() async {
    final shouldDictate = _dictationDecisionWindowOpen();
    if (shouldDictate) {
      if (!_dictating) {
        await _startDictation();
      }
      return;
    }
    if (_dictating) {
      await _stopDictation();
    }
  }

  Future<bool> _prepareAudioChatForSessionStart() async {
    if (_dictating) {
      await _stopDictation();
    }
    if (!_voiceChatActive) {
      setState(() {
        _voiceChatActive = true;
        _voiceStatus = 'Hoert zu';
      });
    }

    try {
      await _ensureVoiceLoopConnected();
    } catch (_) {
      if (!mounted) return false;
      setState(() {
        _voiceChatActive = false;
        _voiceThinking = false;
        _voiceSpeaking = false;
        _voiceStatus = 'Audio-Chat nicht verfuegbar';
      });
      return false;
    }

    // Trigger mic permission/listening directly from the session-start gesture.
    await _startVoiceChatListening();
    if (!_voiceChatActive) {
      return false;
    }

    // Pause during startup turn, resume automatically after first assistant reply.
    await _pauseVoiceRecognitionForTurn();
    return true;
  }

  Future<void> _startPhase4() async {
    if (_phase4Starting || _sending) return;
    final autoEnableVoiceChat = _activeTabIndex == 1;
    var voiceReadyForResume = false;

    if (autoEnableVoiceChat) {
      voiceReadyForResume = await _prepareAudioChatForSessionStart();
      if (!voiceReadyForResume) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Mikrofon-Freigabe fehlt. Bitte Mikrofon zulassen und erneut starten.',
            ),
          ),
        );
        return;
      }
    } else {
      if (_dictating) {
        await _stopDictation();
      }
      if (_voiceChatActive) {
        await _toggleVoiceChat();
      }
    }

    setState(() {
      _phase4Starting = true;
      _phase4Active = true;
      _awaitingDecisionInput = false;
    });

    try {
      final result = await _chatApi.startPhase4(sessionId: _sessionId);
      if (!mounted) return;
      setState(() {
        _messages.clear();
        _inputController.clear();
        _phase4Active = result.phase4Active;
        _awaitingDecisionInput = result.awaitsUserInput;
      });
      _appendMessage(ChatEntry(role: 'assistant', text: result.reply));
      if (autoEnableVoiceChat && _voiceSpeakReady) {
        await _sendSpeakTextOverWs(result.reply, emitText: false);
      } else {
        await _speakAssistantReply(result.reply);
      }
      if (!mounted) return;
      if (autoEnableVoiceChat && voiceReadyForResume) {
        if (!_voiceChatActive || _voiceSpeaking) return;
        await _startVoiceChatListening();
      }
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Fehler beim Start von Phase 4: $error')),
      );
      setState(() {
        _phase4Active = false;
        _awaitingDecisionInput = false;
      });
    } finally {
      if (mounted) {
        setState(() => _phase4Starting = false);
        await _syncDictationGate();
      }
    }
  }

  Future<void> _handleAudioChatPrimaryAction() async {
    if (_phase4Starting || _sending) return;
    if (_voiceChatActive) {
      await _toggleVoiceChat();
      return;
    }
    await _toggleVoiceChat();
  }

  Future<void> _toggleDictation() async {
    if (_dictating) {
      await _stopDictation();
      return;
    }
    if (_voiceChatActive) {
      await _toggleVoiceChat();
    }
    await _startDictation();
  }

  Future<void> _startDictation() async {
    final available = await _speech.initialize(
      onStatus: (status) {
        if (!mounted || !_dictating) return;
        if (status == 'done' || status == 'notListening') {
          setState(() => _dictating = false);
        }
      },
      onError: (_) {
        if (!mounted || !_dictating) return;
        setState(() => _dictating = false);
      },
    );
    if (!available) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Diktieren ist auf diesem GerÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¤t nicht verfÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼gbar.',
          ),
        ),
      );
      return;
    }

    setState(() => _dictating = true);
    await _speech.listen(
      localeId: 'de_CH',
      listenOptions: stt.SpeechListenOptions(
        listenMode: stt.ListenMode.dictation,
        partialResults: true,
        cancelOnError: true,
      ),
      onResult: (result) {
        if (!mounted || !_dictating) return;
        final nextText = result.recognizedWords.trim();
        if (nextText.isEmpty) return;
        setState(() {
          _inputController.value = TextEditingValue(
            text: nextText,
            selection: TextSelection.collapsed(offset: nextText.length),
          );
        });
      },
    );
  }

  Future<void> _stopDictation() async {
    setState(() => _dictating = false);
    try {
      await _speech.stop();
    } catch (_) {}
  }

  Future<void> _toggleVoiceChat() async {
    if (_voiceChatActive) {
      _voiceRestartAllowed = false;
      _voiceRestartTimer?.cancel();
      _voicePartialCommitTimer?.cancel();
      _pendingVoiceTranscript = '';
      _voiceSpeakReady = false;
      setState(() {
        _voiceChatActive = false;
        _voiceThinking = false;
        _voiceSpeaking = false;
        _voiceStatus = 'Bereit';
      });
      try {
        await _speech.stop();
      } catch (_) {}
      try {
        await _assistantAudioPlayer.stop();
      } catch (_) {}
      await _disconnectVoiceLoop(closeSockets: true);
      return;
    }

    if (_dictating) {
      await _stopDictation();
    }

    setState(() {
      _voiceChatActive = true;
      _voiceStatus = 'Verbinde...';
    });
    try {
      await _ensureVoiceLoopConnected();
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _voiceChatActive = false;
        _voiceThinking = false;
        _voiceSpeaking = false;
        _voiceStatus = 'Audio-Chat nicht verfuegbar';
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Audio-Chat Verbindungsfehler: $error')),
      );
      return;
    }
    await _startVoiceChatListening();
  }

  Future<void> _pauseVoiceRecognitionForTurn() async {
    _voiceRestartAllowed = false;
    _voiceRestartTimer?.cancel();
    _voicePartialCommitTimer?.cancel();
    _pendingVoiceTranscript = '';
    try {
      if (_speech.isListening) {
        await _speech.stop();
      }
    } catch (_) {}
  }

  void _queueVoiceTranscript(String text, {required bool finalResult}) {
    final candidate = text.trim();
    if (candidate.isEmpty) return;
    _pendingVoiceTranscript = candidate;
    _voicePartialCommitTimer?.cancel();
    if (finalResult) {
      unawaited(_commitPendingVoiceTranscript());
      return;
    }
    _voicePartialCommitTimer = Timer(const Duration(milliseconds: 900), () {
      unawaited(_commitPendingVoiceTranscript());
    });
  }

  Future<void> _commitPendingVoiceTranscript() async {
    final text = _pendingVoiceTranscript.trim();
    _pendingVoiceTranscript = '';
    _voicePartialCommitTimer?.cancel();
    if (text.isEmpty) return;
    if (!_voiceChatActive || _sending || _voiceSpeaking || !mounted) return;

    final now = DateTime.now();
    if (_lastVoiceTurnText == text &&
        _lastVoiceTurnAt != null &&
        now.difference(_lastVoiceTurnAt!) < const Duration(seconds: 2)) {
      return;
    }
    _lastVoiceTurnText = text;
    _lastVoiceTurnAt = now;
    if (_voiceChatActive && _voiceSpeakReady && _voiceListenChannel != null) {
      await _sendUserTextOverWs(text);
      return;
    }
    await _sendMessage(text, fromVoice: true);
  }

  Future<void> _startVoiceChatListening() async {
    if (!_voiceChatActive || _sending || _voiceSpeaking) return;
    if (_voiceListenChannel == null || !_voiceSpeakReady) {
      try {
        await _ensureVoiceLoopConnected();
      } catch (_) {
        if (!mounted) return;
        setState(() {
          _voiceChatActive = false;
          _voiceStatus = 'Audio-Chat nicht verfuegbar';
        });
        return;
      }
    }

    _voiceRestartAllowed = true;
    final available = await _speech.initialize(
      onStatus: (status) {
        if (!_voiceChatActive || !mounted) return;
        if ((status == 'done' || status == 'notListening') &&
            _voiceRestartAllowed &&
            !_sending &&
            !_voiceSpeaking) {
          if (_pendingVoiceTranscript.trim().isNotEmpty) {
            unawaited(_commitPendingVoiceTranscript());
            return;
          }
          _scheduleVoiceRestart();
        }
      },
      onError: (_) {
        if (!_voiceChatActive || !mounted) return;
        _scheduleVoiceRestart();
      },
    );
    if (!available) {
      if (!mounted) return;
      setState(() {
        _voiceChatActive = false;
        _voiceStatus = 'Audio-Chat nicht verfÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼gbar';
      });
      return;
    }

    if (!mounted) return;
    setState(() => _voiceStatus = 'Hoert zu');

    try {
      await _speech.listen(
        localeId: 'de-DE',
        listenOptions: stt.SpeechListenOptions(
          listenMode: stt.ListenMode.dictation,
          partialResults: true,
          cancelOnError: true,
        ),
        onResult: (result) {
          if (!_voiceChatActive || !mounted) return;
          final text = result.recognizedWords.trim();
          if (text.isEmpty) return;
          _queueVoiceTranscript(text, finalResult: result.finalResult);
        },
      );
    } catch (_) {
      _scheduleVoiceRestart();
    }
  }

  void _scheduleVoiceRestart() {
    _voiceRestartTimer?.cancel();
    if (!_voiceChatActive ||
        !_voiceRestartAllowed ||
        _sending ||
        _voiceSpeaking) {
      return;
    }
    _voiceRestartTimer = Timer(const Duration(milliseconds: 300), () {
      if (!_voiceChatActive ||
          !_voiceRestartAllowed ||
          _sending ||
          _voiceSpeaking) {
        return;
      }
      unawaited(_startVoiceChatListening());
    });
  }

  Uri _buildVoiceWsUri(String path) {
    final base = Uri.parse(_backendBaseUrl);
    final scheme = base.scheme == 'https' ? 'wss' : 'ws';
    final segments = <String>[
      ...base.pathSegments.where((segment) => segment.isNotEmpty),
      'audio',
      path,
    ];
    return Uri(
      scheme: scheme,
      host: base.host,
      port: (base.hasPort && base.port != 0) ? base.port : null,
      pathSegments: segments,
      queryParameters: <String, String>{'session_id': _sessionId},
    );
  }

  Future<void> _ensureVoiceLoopConnected() async {
    if (_voiceSpeakChannel != null &&
        _voiceListenChannel != null &&
        _voiceSpeakReady) {
      return;
    }
    await _disconnectVoiceLoop(closeSockets: true);
    await _connectVoiceSpeak();
    await _connectVoiceListen();
    if (!mounted) return;
    setState(() => _voiceStatus = 'Hoert zu');
  }

  Future<void> _connectVoiceSpeak() async {
    final uri = _buildVoiceWsUri('speak');
    final channel = WebSocketChannel.connect(uri);
    _voiceSpeakChannel = channel;
    _voiceSpeakReady = false;
    final readyCompleter = Completer<void>();
    _voiceSpeakReadyCompleter = readyCompleter;

    _voiceSpeakSubscription = channel.stream.listen(
      _handleVoiceSpeakMessage,
      onError: (Object error) {
        _voiceSpeakReady = false;
        _voiceSpeakChannel = null;
        final completer = _voiceSpeakReadyCompleter;
        if (completer != null && !completer.isCompleted) {
          completer.completeError(error);
        }
        _scheduleVoiceReconnect();
      },
      onDone: () {
        _voiceSpeakReady = false;
        _voiceSpeakChannel = null;
        final completer = _voiceSpeakReadyCompleter;
        if (completer != null && !completer.isCompleted) {
          completer.completeError(Exception('speak websocket closed'));
        }
        _scheduleVoiceReconnect();
      },
      cancelOnError: true,
    );

    _voiceSpeakPingTimer?.cancel();
    _voiceSpeakPingTimer = Timer.periodic(const Duration(seconds: 20), (_) {
      _sendJsonOverChannel(_voiceSpeakChannel, <String, dynamic>{
        'type': 'ping',
      });
    });

    try {
      await readyCompleter.future.timeout(const Duration(seconds: 7));
    } finally {
      if (identical(_voiceSpeakReadyCompleter, readyCompleter)) {
        _voiceSpeakReadyCompleter = null;
      }
    }
  }

  Future<void> _connectVoiceListen() async {
    final uri = _buildVoiceWsUri('listen');
    final channel = WebSocketChannel.connect(uri);
    _voiceListenChannel = channel;
    _voiceListenSubscription = channel.stream.listen(
      _handleVoiceListenMessage,
      onError: (Object error) {
        _voiceListenChannel = null;
        _scheduleVoiceReconnect();
      },
      onDone: () {
        _voiceListenChannel = null;
        _scheduleVoiceReconnect();
      },
      cancelOnError: true,
    );

    _voiceListenPingTimer?.cancel();
    _voiceListenPingTimer = Timer.periodic(const Duration(seconds: 15), (_) {
      _sendJsonOverChannel(_voiceListenChannel, <String, dynamic>{
        'type': 'ping',
      });
    });
  }

  void _scheduleVoiceReconnect() {
    if (_voiceReconnectScheduled || !_voiceChatActive) return;
    _voiceReconnectScheduled = true;
    Future<void>.delayed(const Duration(milliseconds: 600), () async {
      _voiceReconnectScheduled = false;
      if (!_voiceChatActive) return;
      try {
        await _ensureVoiceLoopConnected();
      } catch (_) {}
    });
  }

  Future<void> _disconnectVoiceLoop({required bool closeSockets}) async {
    _voiceSpeakPingTimer?.cancel();
    _voiceSpeakPingTimer = null;
    _voiceListenPingTimer?.cancel();
    _voiceListenPingTimer = null;
    _voiceSpeakReady = false;
    _voiceSpeakReadyCompleter = null;
    _voiceIncomingAudioMessageId = null;
    _voiceIncomingAudioBuffer.takeBytes();
    _voiceAudioQueue.clear();

    await _voiceSpeakSubscription?.cancel();
    _voiceSpeakSubscription = null;
    await _voiceListenSubscription?.cancel();
    _voiceListenSubscription = null;

    if (closeSockets) {
      try {
        await _voiceSpeakChannel?.sink.close();
      } catch (_) {}
      try {
        await _voiceListenChannel?.sink.close();
      } catch (_) {}
    }
    _voiceSpeakChannel = null;
    _voiceListenChannel = null;
  }

  void _sendJsonOverChannel(
    WebSocketChannel? channel,
    Map<String, dynamic> payload,
  ) {
    if (channel == null) return;
    try {
      channel.sink.add(jsonEncode(payload));
    } catch (_) {}
  }

  Future<void> _sendUserTextOverWs(String text) async {
    final channel = _voiceListenChannel;
    if (channel == null || !_voiceSpeakReady) {
      await _sendMessage(text, fromVoice: true);
      return;
    }
    _appendMessage(ChatEntry(role: 'user', text: text));
    if (mounted) {
      setState(() {
        _voiceThinking = true;
        _voiceStatus = 'Warte';
        _awaitingDecisionInput = false;
      });
    }
    _sendJsonOverChannel(channel, <String, dynamic>{
      'type': 'user_text',
      'text': text,
      'session_id': _sessionId,
    });
  }

  Future<void> _sendSpeakTextOverWs(
    String text, {
    required bool emitText,
  }) async {
    final channel = _voiceSpeakChannel;
    if (channel == null || !_voiceSpeakReady) {
      await _speakAssistantReply(text);
      return;
    }
    _sendJsonOverChannel(channel, <String, dynamic>{
      'type': 'speak_text',
      'text': text,
      'emit_text': emitText,
    });
  }

  Map<String, dynamic>? _decodeWsPayload(dynamic message) {
    if (message == null || message is List<int>) {
      return null;
    }
    try {
      final decoded = jsonDecode(message as String);
      if (decoded is Map<String, dynamic>) return decoded;
      if (decoded is Map) return Map<String, dynamic>.from(decoded);
    } catch (_) {
      return null;
    }
    return null;
  }

  void _handleVoiceListenMessage(dynamic message) {
    final payload = _decodeWsPayload(message);
    if (payload == null) return;
    final type = payload['type']?.toString();
    if (type == 'state') {
      final status = payload['status']?.toString() ?? '';
      if (!mounted) return;
      setState(() {
        if (status == 'listening') {
          _voiceStatus = 'Hoert zu';
        } else if (status == 'processing') {
          _voiceStatus = 'Warte';
        } else if (status == 'speaking') {
          _voiceStatus = 'Spricht';
        } else {
          _voiceStatus = 'Bereit';
        }
      });
      return;
    }
    if (type == 'error') {
      final text = payload['message']?.toString().trim() ?? '';
      if (text.isEmpty || !mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
    }
  }

  void _handleVoiceSpeakMessage(dynamic message) {
    final payload = _decodeWsPayload(message);
    if (payload == null) return;
    final type = payload['type']?.toString();
    switch (type) {
      case 'ready':
        _voiceSpeakReady = true;
        final completer = _voiceSpeakReadyCompleter;
        if (completer != null && !completer.isCompleted) {
          completer.complete();
        }
        break;
      case 'state':
        if (!mounted) return;
        final status = payload['status']?.toString() ?? '';
        setState(() {
          _voiceThinking = status == 'processing';
          if (status == 'listening') {
            _voiceStatus = 'Hoert zu';
          } else if (status == 'processing') {
            _voiceStatus = 'Warte';
          } else if (status == 'speaking') {
            _voiceStatus = 'Spricht';
          } else {
            _voiceStatus = _voiceChatActive ? 'Bereit' : 'Bereit';
          }
        });
        break;
      case 'assistant_text':
        final text = payload['text']?.toString().trim() ?? '';
        if (text.isEmpty) return;
        if (mounted) {
          setState(() {
            _voiceThinking = false;
            _phase4Active = payload['phase4_active'] == true || _phase4Active;
            _awaitingDecisionInput = payload['awaits_user_input'] == true;
          });
        }
        if (_messages.isNotEmpty) {
          final last = _messages.last;
          if (!last.isUser && last.text.trim() == text) {
            return;
          }
        }
        _appendMessage(ChatEntry(role: 'assistant', text: text));
        break;
      case 'audio':
        _handleVoiceAudioChunk(payload);
        break;
      case 'stop':
        unawaited(_assistantAudioPlayer.stop());
        break;
      case 'error':
        final text = payload['message']?.toString().trim() ?? '';
        if (text.isNotEmpty && mounted) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text(text)));
        }
        break;
      default:
        break;
    }
  }

  void _handleVoiceAudioChunk(Map<String, dynamic> payload) {
    final rawData = payload['data']?.toString() ?? '';
    if (rawData.isEmpty) return;
    Uint8List bytes;
    try {
      bytes = base64Decode(rawData);
    } catch (_) {
      return;
    }
    final seq = payload['seq'];
    final hasSeq = seq is num;
    final isFinal = payload['final'] == true;
    final messageId = payload['message_id']?.toString();

    if (!hasSeq) {
      _queueVoiceAudio(_VoiceAudioChunk(bytes: bytes, messageId: messageId));
      return;
    }

    final int currentSeq = seq.toInt();
    if (currentSeq == 0 || messageId != _voiceIncomingAudioMessageId) {
      _voiceIncomingAudioBuffer.takeBytes();
      _voiceIncomingAudioMessageId = messageId;
    }
    _voiceIncomingAudioBuffer.add(bytes);
    if (!isFinal) {
      return;
    }
    final assembled = _voiceIncomingAudioBuffer.takeBytes();
    _queueVoiceAudio(_VoiceAudioChunk(bytes: assembled, messageId: messageId));
    _voiceIncomingAudioMessageId = null;
  }

  void _queueVoiceAudio(_VoiceAudioChunk chunk) {
    if (chunk.bytes.isEmpty) return;
    _voiceAudioQueue.add(chunk);
    if (_voiceAudioQueueRunning) return;
    unawaited(_drainVoiceAudioQueue());
  }

  Future<void> _drainVoiceAudioQueue() async {
    if (_voiceAudioQueueRunning) return;
    _voiceAudioQueueRunning = true;
    try {
      while (_voiceAudioQueue.isNotEmpty && _voiceChatActive) {
        final chunk = _voiceAudioQueue.removeFirst();
        if (mounted) {
          setState(() {
            _voiceSpeaking = true;
            _voiceThinking = false;
            _voiceStatus = 'Spricht';
          });
        }
        await _pauseVoiceRecognitionForTurn();
        await _assistantAudioPlayer.stop();
        await _assistantAudioPlayer.play(BytesSource(chunk.bytes), volume: 1.0);
        await _assistantAudioPlayer.onPlayerComplete.first.timeout(
          const Duration(minutes: 5),
        );
      }
    } catch (_) {
    } finally {
      _voiceAudioQueueRunning = false;
      final shouldRestartListening = mounted && _voiceChatActive;
      if (mounted) {
        setState(() {
          _voiceSpeaking = false;
          _voiceStatus = _voiceChatActive ? 'Hoert zu' : 'Bereit';
        });
      }
      if (shouldRestartListening) {
        await _startVoiceChatListening();
      }
    }
  }

  void _changeTextScale({required bool increase}) {
    final delta = increase ? 0.05 : -0.05;
    final next = (_chatTextScale + delta).clamp(0.85, 1.35);
    if ((next - _chatTextScale).abs() < 0.0001) return;
    setState(() => _chatTextScale = next);
  }

  Future<void> _setActiveTab(int index) async {
    if (index == _activeTabIndex) return;
    if (index == 1 && _dictating) {
      await _stopDictation();
    }
    if (!mounted) return;
    setState(() => _activeTabIndex = index);
    await _syncDictationGate();
  }

  Widget _buildModeTabs() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: Container(
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          color: const Color(0xFF111A2B),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFF3D4D70)),
        ),
        child: Row(
          children: [
            Expanded(child: _buildModeTabButton(index: 0, label: 'Chat')),
            Expanded(child: _buildModeTabButton(index: 1, label: 'Audio-Chat')),
          ],
        ),
      ),
    );
  }

  Widget _buildModeTabButton({required int index, required String label}) {
    final active = _activeTabIndex == index;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () => unawaited(_setActiveTab(index)),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            gradient: active
                ? const LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [Color(0xFF3C75FF), Color(0xFF2DA0FF)],
                  )
                : null,
            color: active ? null : const Color(0xFF1A2438),
          ),
          child: Text(
            label,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withValues(alpha: active ? 0.95 : 0.78),
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPhase4ControlRow() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: Row(
        children: [
          Expanded(
            child: OutlinedButton.icon(
              onPressed: _phase4Starting ? null : _startPhase4,
              icon: Icon(
                _phase4Starting ? Icons.hourglass_bottom : Icons.play_arrow,
              ),
              label: Text(
                _phase4Starting
                    ? 'Phase 4 startet ...'
                    : (_phase4Active
                          ? 'Phase 4 neu starten'
                          : 'Phase 4 starten'),
              ),
            ),
          ),
          if (_phase4Active) ...[
            const SizedBox(width: 10),
            Text(
              _awaitingDecisionInput ? 'Frage offen' : 'KI aktiv',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildAudioChatPanel() {
    final canToggle = !_sending && !_phase4Starting;
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF121B2D),
        borderRadius: BorderRadius.circular(25),
        border: Border.all(color: const Color(0xFF59607A)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x66000000),
            blurRadius: 16,
            offset: Offset(0, 5),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Audio-Chat Testmodus',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.9),
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Hier ist nur Audio-Chat aktiv. Sage danach z.B. "Starte jetzt mit der Hypnose Session", dann startet Phase 4.',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.7),
              fontSize: 12.5,
              height: 1.3,
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: canToggle ? _handleAudioChatPrimaryAction : null,
            icon: Icon(
              _voiceChatActive ? Icons.stop_circle : Icons.multitrack_audio,
            ),
            label: Text(
              _voiceChatActive ? 'Audio-Chat stoppen' : 'Audio-Chat starten',
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final canSend =
        _inputController.text.trim().isNotEmpty &&
        !_sending &&
        !_phase4Starting &&
        (!_phase4Active || _awaitingDecisionInput);
    final isAudioTab = _activeTabIndex == 1;

    return Scaffold(
      body: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF0E172B), Color(0xFF070C18)],
          ),
        ),
        child: SafeArea(
          child: Stack(
            children: [
              Column(
                children: [
                  const SizedBox(height: 10),
                  _buildModeTabs(),
                  if (!isAudioTab) _buildPhase4ControlRow(),
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 14),
                      child: _buildMessageList(),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                    child: isAudioTab
                        ? _buildAudioChatPanel()
                        : _buildInputBar(canSend: canSend),
                  ),
                ],
              ),
              if (_voiceChatActive)
                Positioned(
                  right: 26,
                  bottom: isAudioTab ? 150 : 120,
                  child: _AudioStatusBadge(
                    status: _voiceStatus,
                    speaking: _voiceSpeaking,
                    thinking: _voiceThinking,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMessageList() {
    if (_messages.isEmpty && !_sending) {
      return Center(
        child: Text(
          'Startklar. Sprich oder schreibe deine Nachricht.',
          textAlign: TextAlign.center,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.72),
            fontSize: 16 * _chatTextScale,
          ),
        ),
      );
    }

    final itemCount = _messages.length + (_sending ? 1 : 0);
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.only(top: 14, bottom: 20),
      itemCount: itemCount,
      itemBuilder: (context, index) {
        final isTypingBubble = _sending && index == itemCount - 1;
        if (isTypingBubble) {
          return const SizedBox.shrink();
        }

        final entry = _messages[index];
        final align = entry.isUser
            ? Alignment.centerRight
            : Alignment.centerLeft;
        final bubbleColor = entry.isUser
            ? const Color(0xFF2A5FD4)
            : const Color(0xFF1A2436);
        return Align(
          alignment: align,
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 480),
            child: Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              decoration: BoxDecoration(
                color: bubbleColor,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                entry.text,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 15 * _chatTextScale,
                  height: 1.35,
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildInputBar({required bool canSend}) {
    final dictationAllowed =
        !_phase4Active || _dictationDecisionWindowOpen() || _dictating;
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF121B2D),
        borderRadius: BorderRadius.circular(25),
        border: Border.all(color: const Color(0xFF59607A)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x66000000),
            blurRadius: 16,
            offset: Offset(0, 5),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: _inputController,
            focusNode: _inputFocusNode,
            minLines: 1,
            maxLines: 7,
            keyboardType: TextInputType.multiline,
            textInputAction: TextInputAction.send,
            style: const TextStyle(color: Colors.white, fontSize: 18),
            cursorColor: const Color(0xFF6E9BFF),
            decoration: const InputDecoration(
              border: InputBorder.none,
              hintText: 'Nachricht an AIVORA...',
              hintStyle: TextStyle(color: Color(0xFFB8B7BD)),
            ),
            onChanged: (_) => setState(() {}),
            onSubmitted: (_) => _sendFromInput(),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              _CircleActionButton(
                icon: Icons.remove,
                onTap: () => _changeTextScale(increase: false),
                active: false,
                compact: true,
              ),
              _CircleActionButton(
                icon: Icons.add,
                onTap: () => _changeTextScale(increase: true),
                active: false,
                compact: true,
              ),
              const SizedBox(width: 8),
              Text(
                'Schrift',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.86),
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              _CircleActionButton(
                icon: Icons.mic,
                onTap: dictationAllowed ? _toggleDictation : null,
                active: _dictating || (_phase4Active && _awaitingDecisionInput),
              ),
              _CircleActionButton(
                icon: _sending ? Icons.hourglass_bottom : Icons.send,
                onTap: canSend ? _sendFromInput : null,
                active: canSend,
              ),
            ],
          ),
          if (_phase4Active)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                _awaitingDecisionInput
                    ? 'Frageknoten offen: Diktieren ist aktiv.'
                    : 'KI spricht/verarbeitet: Diktieren ist inaktiv.',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.65),
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _CircleActionButton extends StatelessWidget {
  const _CircleActionButton({
    required this.icon,
    required this.onTap,
    required this.active,
    this.compact = false,
  });

  final IconData icon;
  final FutureOr<void> Function()? onTap;
  final bool active;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final size = compact ? 38.0 : 46.0;
    final iconSize = compact ? 18.0 : 22.0;
    final gradient = active
        ? const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF3C75FF), Color(0xFF2DA0FF)],
          )
        : const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF23324B), Color(0xFF1A2438)],
          );

    return Container(
      margin: EdgeInsets.symmetric(
        horizontal: compact ? 2 : 4,
        vertical: compact ? 4 : 6,
      ),
      decoration: BoxDecoration(
        gradient: gradient,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onTap == null
              ? null
              : () async {
                  await Future<void>.value(onTap!());
                },
          child: SizedBox(
            width: size,
            height: size,
            child: Icon(
              icon,
              size: iconSize,
              color: onTap == null
                  ? Colors.white.withValues(alpha: 0.32)
                  : Colors.white.withValues(alpha: 0.92),
            ),
          ),
        ),
      ),
    );
  }
}

class _AudioStatusBadge extends StatelessWidget {
  const _AudioStatusBadge({
    required this.status,
    required this.speaking,
    required this.thinking,
  });

  final String status;
  final bool speaking;
  final bool thinking;

  @override
  Widget build(BuildContext context) {
    final color = speaking
        ? const Color(0xFF4A7AFF)
        : (thinking ? const Color(0xFF7182A7) : const Color(0xFF2B3E5D));
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(14),
        boxShadow: const [
          BoxShadow(
            color: Color(0x66000000),
            blurRadius: 14,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            speaking
                ? Icons.volume_up
                : (thinking ? Icons.bolt : Icons.hearing),
            color: Colors.white,
            size: 16,
          ),
          const SizedBox(width: 8),
          Text(
            status,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _VoiceAudioChunk {
  const _VoiceAudioChunk({required this.bytes, this.messageId});

  final Uint8List bytes;
  final String? messageId;
}

class ChatEntry {
  const ChatEntry({required this.role, required this.text});

  final String role;
  final String text;

  bool get isUser => role == 'user';
}

class ChatApiReply {
  const ChatApiReply({
    required this.reply,
    required this.awaitsUserInput,
    required this.phase4Active,
  });

  final String reply;
  final bool awaitsUserInput;
  final bool phase4Active;
}

class ChatApiClient {
  ChatApiClient({required this.baseUrl, http.Client? client})
    : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Future<ChatApiReply> sendMessage({
    required String sessionId,
    required String message,
  }) async {
    final uri = Uri.parse('${baseUrl.replaceAll(RegExp(r"/+$"), "")}/chat');
    final response = await _client
        .post(
          uri,
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode(<String, dynamic>{
            'session_id': sessionId,
            'message': message,
          }),
        )
        .timeout(const Duration(seconds: 45));
    return _decodeReply(response);
  }

  Future<ChatApiReply> startPhase4({required String sessionId}) async {
    final uri = Uri.parse(
      '${baseUrl.replaceAll(RegExp(r"/+$"), "")}/phase4/start',
    );
    final response = await _client
        .post(
          uri,
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode(<String, dynamic>{'session_id': sessionId}),
        )
        .timeout(const Duration(seconds: 45));
    return _decodeReply(response);
  }

  Future<Uint8List> requestTtsAudio({
    required String text,
    required String context,
    int? phase,
  }) async {
    final uri = Uri.parse(
      '${baseUrl.replaceAll(RegExp(r"/+$"), "")}/api/tts-audio',
    );
    final response = await _client
        .post(
          uri,
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode(<String, dynamic>{
            'text': text,
            'context': context,
            if (phase != null) 'phase': phase,
          }),
        )
        .timeout(const Duration(seconds: 75));
    if (response.statusCode >= 400) {
      final rawBody = response.body;
      String detail = 'HTTP ${response.statusCode}';
      if (rawBody.isNotEmpty) {
        try {
          final decoded = jsonDecode(rawBody) as Map<String, dynamic>;
          detail = decoded['detail']?.toString() ?? detail;
        } catch (_) {
          detail = rawBody;
        }
      }
      throw Exception(detail);
    }
    final bytes = response.bodyBytes;
    if (bytes.isEmpty) {
      throw Exception('Leere TTS-Antwort vom Backend.');
    }
    return bytes;
  }

  ChatApiReply _decodeReply(http.Response response) {
    final rawBody = response.body;
    final decoded = rawBody.isEmpty
        ? const <String, dynamic>{}
        : (jsonDecode(rawBody) as Map<String, dynamic>);
    if (response.statusCode >= 400) {
      final detail = decoded['detail']?.toString();
      throw Exception(detail ?? 'HTTP ${response.statusCode}');
    }

    final reply = decoded['reply']?.toString().trim() ?? '';
    if (reply.isEmpty) {
      throw Exception('Leere Antwort vom Backend.');
    }
    return ChatApiReply(
      reply: reply,
      awaitsUserInput: decoded['awaits_user_input'] == true,
      phase4Active: decoded['phase4_active'] == true,
    );
  }

  void dispose() {
    _client.close();
  }
}
