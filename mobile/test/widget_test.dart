// Basic smoke test for Phase 0: confirms the app boots to the login
// screen, and that logging in navigates to the (empty) case list.

import 'package:flutter_test/flutter_test.dart';

import 'package:pro_se_guardian/main.dart';

void main() {
  testWidgets('App starts on login screen', (WidgetTester tester) async {
    await tester.pumpWidget(const ProSeGuardianApp());

    expect(find.text('Pro Se Guardian AI'), findsOneWidget);
    expect(find.text('Log In'), findsOneWidget);
  });

  testWidgets('Logging in shows empty case list', (WidgetTester tester) async {
    await tester.pumpWidget(const ProSeGuardianApp());

    await tester.tap(find.text('Log In'));
    await tester.pumpAndSettle();

    expect(find.text('Your Cases'), findsOneWidget);
    expect(find.textContaining('No cases yet'), findsOneWidget);
  });
}
