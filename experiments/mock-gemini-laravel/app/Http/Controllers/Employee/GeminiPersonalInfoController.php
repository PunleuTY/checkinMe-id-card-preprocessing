<?php

namespace App\Http\Controllers\Employee;

use Carbon\Carbon;
use Illuminate\Http\Request;
use App\Services\GeminiService;
use App\Http\Controllers\Controller;

class GeminiPersonalInfoController extends Controller
{
    public static function readIdCard(Request $request)
    {
        $idCard = new GeminiService();
        $response = $idCard->scanNid($request->idImage);

        if (!$response['success']) {
            return fail($response['message']);
        }

        return success([
            'name' => $response['data']['lastNameKh'] . ' ' . $response['data']['firstNameKh'],
            'name_en' => $response['data']['lastNameEn'] . ' ' . $response['data']['firstNameEn'],
            'gender' => $response['data']['gender'] == 'M' ? 'male' : 'female',
            'nationality' => 'khmer',
            'address' => $response['data']['address'],
            'dob' => Carbon::createFromFormat('d/m/Y', $response['data']['dob'])->format('Y-m-d'),
            'dob_display' => Carbon::createFromFormat('d/m/Y', $response['data']['dob'])->format('d F Y'),
            'id_card_number' => $response['data']['idNumber'],
            'id_card_expire_date' => Carbon::createFromFormat('d/m/Y', $response['data']['expiredDate'])->format('Y-m-d'),
            'id_card_expire_date_display' => Carbon::createFromFormat('d/m/Y', $response['data']['expiredDate'])->format('d F Y')
        ]);
    }
}