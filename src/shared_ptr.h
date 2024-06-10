// a minimalistic implementation of shared_ptr adapted from boost v1.27
// limitations: no weak_ptr support, not thread-safe
#pragma once
#include <algorithm>

namespace shared_ptr_internal{

class counted_base {
public:
    explicit counted_base(long initial_use_count) : use_count_(initial_use_count) {}

    virtual ~counted_base() {}  // nothrow

    // dispose() is called when use_count_ drops to zero, to release
    // the resources managed by *this.
    //
    // counted_base doesn't manage any resources except itself, and
    // the default implementation is a no-op.

    virtual void dispose() = 0; // nothrow

    void add_ref() // nothrow
    {
        ++use_count_;
    }

    void release() // nothrow
    {
        if(--use_count_ == 0)
        {
            dispose();
            delete this;
        }
    }

private:
    counted_base(counted_base const &);
    counted_base & operator= (counted_base const &);

    long use_count_;
};


template<class P, class D> class counted_base_impl: public counted_base {
private:
    P ptr; // copy constructor must not throw
    D del; // copy constructor must not throw

    counted_base_impl(counted_base_impl const &);
    counted_base_impl & operator= (counted_base_impl const &);

public:
    counted_base_impl(P p, D d, long initial_use_count) : counted_base(initial_use_count), ptr(p), del(d) {}

    virtual void dispose() // nothrow
    {
        del(ptr);
    }
};


class shared_count {
private:
    counted_base * pi_;

public:

    template<class P, class D> shared_count(P p, D d): pi_(0)
    {
        try {
            pi_ = new counted_base_impl<P, D>(p, d, 1);
        }
        catch(...) {
            d(p); // delete p
            throw;
        }
    }

    ~shared_count() // nothrow
    {
        if(pi_ != 0) pi_->release();
    }

    shared_count(shared_count const & r): pi_(r.pi_) // nothrow
    {
        if(pi_ != 0) pi_->add_ref();
    }

    shared_count & operator= (shared_count const & r) // nothrow
    {
        counted_base * tmp = r.pi_;
        if(tmp != 0) tmp->add_ref();
        if(pi_ != 0) pi_->release();
        pi_ = tmp;
        return *this;
    }

    void swap(shared_count & r) // nothrow
    {
        counted_base * tmp = r.pi_;
        r.pi_ = pi_;
        pi_ = tmp;
    }
};

}  // namespace shared_ptr_internal

template<typename T> class shared_ptr {
private:
    T * px;  // contained pointer
    shared_ptr_internal::shared_count pn;  // reference counter
    template<typename Y> friend class shared_ptr;

    struct checked_deleter {
        typedef void result_type;
        typedef T * argument_type;
        void operator()(T * x)
        {
            typedef char type_must_be_complete[sizeof(T)];
            (void)sizeof(type_must_be_complete);
            delete x;
        }
    };

public:

    explicit shared_ptr(T * p = 0): px(p), pn(p, checked_deleter()) {}

    template<typename D> shared_ptr(T * p, D d): px(p), pn(p, d) {}

    template<typename Y> shared_ptr(const shared_ptr<Y>& r): px(r.px), pn(r.pn) {}

    template<typename Y> shared_ptr(const shared_ptr<Y>& r, T* p): px(p), pn(r.pn) {}  // aliasing constructor

    template<typename Y> shared_ptr & operator=(shared_ptr<Y> const & r) {
        px = r.px;
        pn = r.pn; // shared_count::op= doesn't throw
        return *this;
    }

    void reset(T * p = 0)
    {
        if(px == p) return;
        shared_ptr<T>(p).swap(*this);
    }

    T& operator*()  const { return *px; }
    T* operator->() const { return px; }
    T* get()        const { return px; }
    //operator T*()   const { return px; }
    typedef T * (shared_ptr<T>::*unspecified_bool_type)() const;
    operator unspecified_bool_type() const // never throws
    {
        return px == 0 ? 0: &shared_ptr<T>::get;
    }

    void swap(shared_ptr<T> & other) // never throws
    {
        std::swap(px, other.px);
        pn.swap(other.pn);
    }

};  // shared_ptr

template<typename T> void swap(shared_ptr<T> & a, shared_ptr<T> & b)
{
    a.swap(b);
}
