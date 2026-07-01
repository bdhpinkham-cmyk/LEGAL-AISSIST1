import 'package:flutter/material.dart';
import 'new_case_screen.dart';

/// Home screen: the case list. Phase 0: no backend wiring yet, so this
/// list is always empty (a hardcoded empty list). Task 4 wires this up
/// to real Supabase data.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // Placeholder — replaced with real Supabase-backed case data in Task 4.
  final List<String> _cases = [];

  void _goToNewCase() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => const NewCaseScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Your Cases')),
      body: _cases.isEmpty
          ? const Center(
              child: Padding(
                padding: EdgeInsets.all(24.0),
                child: Text(
                  'No cases yet.\nTap "New Case" to add one.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.grey),
                ),
              ),
            )
          : ListView.builder(
              itemCount: _cases.length,
              itemBuilder: (context, index) => ListTile(
                title: Text(_cases[index]),
              ),
            ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _goToNewCase,
        icon: const Icon(Icons.add),
        label: const Text('New Case'),
      ),
    );
  }
}
