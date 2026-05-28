<?php

return [
    'base_url' => env('GEMINI_SERVICE_BASE_URL', 'http://localhost:8000'),
    'timeout' => env('GEMINI_SERVICE_TIMEOUT', 30),
    'model' => env('GEMINI_SERVICE_MODEL', 'gemini-2.5-flash'),
    'endpoints' => [
        'scan_nid' => env('GEMINI_SERVICE_ENDPOINTS_SCAN_NID', '/gemini-ocr/upload'),
    ]
];

