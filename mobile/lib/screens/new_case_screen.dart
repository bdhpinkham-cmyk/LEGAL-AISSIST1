import 'package:flutter/material.dart';

/// New Case screen — placeholder only for Phase 0, Task 2.
/// Task 4 turns this into a real form that inserts a row into the
/// Supabase `cases` table.
class NewCaseScreen extends StatelessWidget {
  const NewCaseScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('New Case')),
      body: const Center(
        child: Padding(
          padding: EdgeInsets.all(24.0),
          child: Text(
            'Case form coming in Phase 0, Task 4\n'
            '(case name, jurisdiction, charges).',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16, color: Colors.grey),
          ),
        ),
      ),
    );
  }
}
