<?php

// Trimmed from dev env app/Helpers/helper.php.
// Only the response helpers used by GeminiReadIdCardController and GeminiPersonalInfoController are kept.
// The dev env version also references ApiAccessLog in fail() to log failed API calls — omitted here
// since the mock has no API access logging infrastructure.

function success($data = null)
{
    return response()->json(["success" => true, "data" => $data]);
}

function fail($message = null, $code = 400, $data = null)
{
    return response()->json(["success" => false, "message" => $message, "data" => $data], $code);
}
