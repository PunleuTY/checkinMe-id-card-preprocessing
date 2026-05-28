<?php

namespace App\Http\Controllers\Client;

use Illuminate\Http\Request;
use App\Services\GeminiService;
use App\Http\Controllers\Controller;

class GeminiReadIdCardController extends Controller
{
    public static function readIdCard(Request $request)
    {
        $idCard = new GeminiService();
        $response = $idCard->scanNid($request->idImage);

        if (!$response['success']) {
            return fail($response['message']);
        }

        return success($response['data']);
    }
}