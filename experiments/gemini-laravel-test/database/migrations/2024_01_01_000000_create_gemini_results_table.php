<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

// Modeled after dev env: database/migrations/2023_09_18_113531_create_camdx_results_table.php
// Table name changed from camdx_results → gemini_results to match GeminiResult model.
// Schema is identical — both services log request/response pairs with polymorphic creator tracking.

class CreateGeminiResultsTable extends Migration
{
    public function up()
    {
        Schema::create('gemini_results', function (Blueprint $table) {
            $table->id();
            $table->string('status_code')->nullable();
            $table->longText('request_data');
            $table->longText('response_data')->nullable();
            $table->string('created_by_id')->nullable();
            $table->string('created_by_type')->nullable();
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('gemini_results');
    }
}