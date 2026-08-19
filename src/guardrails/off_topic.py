from typing import Tuple

class OffTopicGuardrail:
    """Detects off-topic or out-of-domain queries."""
    
    def evaluate(self, classification: str) -> Tuple[bool, str]:
        if classification == "UNSAFE":
            return False, "மன்னிக்கவும், இந்த கேள்வி பாதுகாப்பற்றது என்பதால் பதில் அளிக்க முடியாது."
        if classification == "OUT_OF_DOMAIN":
            return False, "மன்னிக்கவும், வழங்கப்பட்ட அறிவுக் களஞ்சியத்தில் இந்தத் தலைப்பில் தரவு இல்லை."
        return True, "PASSED"
