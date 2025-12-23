import re

class PIIScrubber:
    """
    detects and redacting Personally Identifiable Information (PII) from text.
    """

    # Compiled regex patterns for performance
    PATTERNS = {
        'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'PHONE': re.compile(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b'),
        'CREDIT_CARD': re.compile(r'\b(?:\d{4}[- ]){3}\d{4}\b|\b\d{16}\b'),
        # SSN (US) - Basic pattern
        'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    }

    @classmethod
    def scrub_text(cls, text: str) -> str:
        """
        Redacts PII from the input text.
        
        Args:
            text: The input string containing potential PII.
            
        Returns:
            The sanitized string with PII replaced by placeholders (e.g., [EMAIL_REDACTED]).
        """
        if not text:
            return text
            
        scrubbed_text = text
        
        # Apply each pattern
        for pii_type, pattern in cls.PATTERNS.items():
            replacement = f"[{pii_type}_REDACTED]"
            scrubbed_text = pattern.sub(replacement, scrubbed_text)
            
        return scrubbed_text

def scrub_text(text: str) -> str:
    """Wrapper function for easier import."""
    return PIIScrubber.scrub_text(text)
