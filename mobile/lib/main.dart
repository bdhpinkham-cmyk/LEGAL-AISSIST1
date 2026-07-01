import 'package:flutter/material.dart';
import 'screens/login_screen.dart';

void main() {
  runApp(const ProSeGuardianApp());
}

/// Root widget. Phase 0: no backend wiring yet — this just sets up the
/// three-screen navigation flow (Login -> Home/case list -> New Case).
class ProSeGuardianApp extends StatelessWidget {
  const ProSeGuardianApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pro Se Guardian AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const LoginScreen(),
    );
  }
}
