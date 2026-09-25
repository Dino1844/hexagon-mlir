//===- hmx-vtcm-event-context-identity-reject.mlir - no cross-sidecar guess --===//
//
// An eligible event context plus a present identity sidecar is one claim about
// one canonical scope. Any disagreement, and any identity record that omits a
// fact the event record states, is refused: picking one side would attribute an
// opaque token to a site no compiler fact supports.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(hmx-to-llvm)'
//===----------------------------------------------------------------------===//

// expected-error @+1 {{does not reconcile with hmx.kernel_vtcm_identity: scope_id}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_identity = {
  functions = [{
    function = "kernel",
    function_id = 101 : i64,
    sites = [{
      event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
      event_token_bits = 128 : i64,
      event_token_high = 405 : i64,
      event_token_low = 404 : i64,
      function = "kernel",
      function_id = 101 : i64,
      identity_status = "complete",
      site_id = 202 : i64
    }]
  }],
  kind = "static-vtcm-site-identity-v1",
  schema = "hmx.vtcm-static-identity/v1",
  scope_id = 999 : i64,
  status = "complete"
},
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  delayed_cache_owner_status = "aggregate",
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  grid_scope_status = "not-proven",
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  performance_claimed = 0 : i64,
  runtime_event_join_status = "not-proven",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// The token is the join key: a different token in the identity record is a
// different site, even though every other fact matches.
// expected-error @+1 {{does not reconcile with hmx.kernel_vtcm_identity: canonical site or event token}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_identity = {
  functions = [{
    function = "kernel",
    function_id = 101 : i64,
    sites = [{
      event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
      event_token_bits = 128 : i64,
      event_token_high = 405 : i64,
      event_token_low = 999 : i64,
      function = "kernel",
      function_id = 101 : i64,
      identity_status = "complete",
      site_id = 202 : i64
    }]
  }],
  kind = "static-vtcm-site-identity-v1",
  schema = "hmx.vtcm-static-identity/v1",
  scope_id = 303 : i64,
  status = "complete"
},
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  delayed_cache_owner_status = "aggregate",
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  grid_scope_status = "not-proven",
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  performance_claimed = 0 : i64,
  runtime_event_join_status = "not-proven",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// A site id without the token words cannot be reconciled: the record claims an
// identity but not the opaque words the event record states.
// expected-error @+1 {{does not reconcile with hmx.kernel_vtcm_identity: canonical site or event token}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_identity = {
  functions = [{
    function = "kernel",
    function_id = 101 : i64,
    sites = [{
      event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
      event_token_bits = 128 : i64,
      event_token_high = 405 : i64,
      function = "kernel",
      function_id = 101 : i64,
      identity_status = "complete",
      site_id = 202 : i64
    }]
  }],
  kind = "static-vtcm-site-identity-v1",
  schema = "hmx.vtcm-static-identity/v1",
  scope_id = 303 : i64,
  status = "complete"
},
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  delayed_cache_owner_status = "aggregate",
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  grid_scope_status = "not-proven",
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  performance_claimed = 0 : i64,
  runtime_event_join_status = "not-proven",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// The event record names a function the identity sidecar does not claim.
// expected-error @+1 {{does not reconcile with hmx.kernel_vtcm_identity: function_id}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_identity = {
  functions = [{
    function = "other",
    function_id = 101 : i64,
    sites = [{
      event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
      event_token_bits = 128 : i64,
      event_token_high = 405 : i64,
      event_token_low = 404 : i64,
      function = "kernel",
      function_id = 101 : i64,
      identity_status = "complete",
      site_id = 202 : i64
    }]
  }],
  kind = "static-vtcm-site-identity-v1",
  schema = "hmx.vtcm-static-identity/v1",
  scope_id = 303 : i64,
  status = "complete"
},
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  delayed_cache_owner_status = "aggregate",
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  grid_scope_status = "not-proven",
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  performance_claimed = 0 : i64,
  runtime_event_join_status = "not-proven",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// Two claimed sites cannot be reconciled with a one-site event record; the
// extra claim is exactly the ambiguity the eligible scope refuses.
// expected-error @+1 {{does not reconcile with hmx.kernel_vtcm_identity: two sites claim an identity}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_identity = {
  functions = [{
    function = "kernel",
    function_id = 101 : i64,
    sites = [
      {
        event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
        event_token_bits = 128 : i64,
        event_token_high = 405 : i64,
        event_token_low = 404 : i64,
        function = "kernel",
        function_id = 101 : i64,
        identity_status = "complete",
        site_id = 202 : i64
      },
      {
        event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
        event_token_bits = 128 : i64,
        event_token_high = 407 : i64,
        event_token_low = 406 : i64,
        function = "kernel",
        function_id = 101 : i64,
        identity_status = "complete",
        site_id = 203 : i64
      }
    ]
  }],
  kind = "static-vtcm-site-identity-v1",
  schema = "hmx.vtcm-static-identity/v1",
  scope_id = 303 : i64,
  status = "complete"
},
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  delayed_cache_owner_status = "aggregate",
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  grid_scope_status = "not-proven",
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  performance_claimed = 0 : i64,
  runtime_event_join_status = "not-proven",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// A present identity attribute of the wrong kind is not a record to reconcile.
// expected-error @+1 {{does not reconcile with hmx.kernel_vtcm_identity: identity sidecar is not a dictionary}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_identity = "static-vtcm-site-identity-v1",
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  delayed_cache_owner_status = "aggregate",
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  grid_scope_status = "not-proven",
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  performance_claimed = 0 : i64,
  runtime_event_join_status = "not-proven",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}
