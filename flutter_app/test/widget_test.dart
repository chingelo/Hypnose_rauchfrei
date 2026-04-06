import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_app/main.dart';

void main() {
  testWidgets('shows chat bootstrap text', (WidgetTester tester) async {
    await tester.pumpWidget(const HypnoseChatApp());
    expect(
      find.text('Startklar. Sprich oder schreibe deine Nachricht.'),
      findsOneWidget,
    );
  });
}
